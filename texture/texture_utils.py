import os.path

from psbody.mesh import Mesh
from psbody.mesh.geometry.barycentric_coordinates_of_projection import barycentric_coordinates_of_projection
from psbody.mesh.visibility import visibility_compute
from psbody.mesh.meshviewer import MeshViewer, MeshViewers
import numpy as np
import scipy
import cv2
from texture.texture_setting import settings, animal_output_dir
import torch


def uv_to_xyz_and_normals(alignment, face_indices_map, b_coords_map):
    if not hasattr(alignment, 'vn'):
        alignment.reset_normals()

    pixels_to_set = np.array(np.where(face_indices_map != -1)).T
    x_to_set = pixels_to_set[:, 0]
    y_to_set = pixels_to_set[:, 1]
    b_coords = b_coords_map[x_to_set, y_to_set, :]
    f_coords = face_indices_map[x_to_set, y_to_set].astype(np.int32)
    v_ids = alignment.f[f_coords]
    points = np.tile(b_coords[:, 0], (3, 1)).T * alignment.v[v_ids[:, 0]] + \
             np.tile(b_coords[:, 1], (3, 1)).T * alignment.v[v_ids[:, 1]] + \
             np.tile(b_coords[:, 2], (3, 1)).T * alignment.v[v_ids[:, 2]]
    normals = np.tile(b_coords[:, 0], (3, 1)).T * alignment.vn[v_ids[:, 0]] + \
              np.tile(b_coords[:, 1], (3, 1)).T * alignment.vn[v_ids[:, 1]] + \
              np.tile(b_coords[:, 2], (3, 1)).T * alignment.vn[v_ids[:, 2]]
    return (points, normals)


def generate_template_map_by_triangles(vt, ft, map_scale=1.):
    # face_indices_map: 一张图表示，上面每个坐标是uv上的真实坐标，其值对应于对应面的索引
    # b_coords_map : 一张图表示，上面记录了face_indices_map有效的点，经过插值之后的uv坐标
    # 本函数的作用是做预处理，生成纹理坐标并存储，后面可以直接查表
    # shape of ft : (7774, 3)
    # input vt & ft : numpy array

    map_height = map_width = int(2048 * map_scale)

    face_indices_map = np.ones((map_height, map_width, 3)) * -1
    text_coords = vt[ft.flatten()][:, :2]
    # compute following formula:
    # absolute text coord : [map_width*text_coord, (1-map_height)*coord]
    text_coords *= np.tile([map_width, -map_height], (ft.size, 1))
    text_coords += np.hstack((np.zeros((ft.size, 1)), map_height * np.ones((ft.size, 1))))
    text_coords_tr = text_coords.reshape(-1, 6).astype(np.int32)
    # XXX fillConvexPoly seems like an overkill for drawing (many) triangles
    # We should either draw them in opengl or convert it to c++
    # 把当前所有属于同一类的三角面片，其坐标统一赋值为一类；按照uv做划分
    for itc, tc in enumerate(text_coords_tr):
        cv2.fillConvexPoly(face_indices_map, tc.reshape(3, 2), [itc, itc, itc])

    face_indices_map = face_indices_map[:, :, 0]

    # 找到对应有效位置 + texture coord
    pixels_to_set = np.array(np.where(face_indices_map != -1)).T

    x_to_set = pixels_to_set[:, 0]
    y_to_set = pixels_to_set[:, 1]
    f_indices = face_indices_map[x_to_set, y_to_set].astype(np.int32)

    # new method
    text_coords = np.reshape(np.fliplr(text_coords), (-1, 6)).astype(np.int32)[f_indices]

    points = np.hstack((pixels_to_set, np.zeros((pixels_to_set.shape[0], 1))))

    first_v = np.hstack((text_coords[:, 0:2], np.zeros((text_coords.shape[0], 1))))
    second_v = np.hstack((text_coords[:, 2:4], np.zeros((text_coords.shape[0], 1))))
    third_v = np.hstack((text_coords[:, 4:6], np.zeros((text_coords.shape[0], 1))))

    b_coords = barycentric_coordinates_of_projection(points, first_v, second_v - first_v, third_v - first_v)
    b_coords_flat = b_coords.flatten()
    wrong_pixels = np.union1d(np.where(b_coords_flat < 0)[0], np.where(b_coords_flat > 1)[0])
    wrong_ids = np.unique(np.floor(wrong_pixels / 3.0).astype(np.int32))

    b_coords_map = np.ones((map_height, map_width, 3)) * -1
    b_coords_map[x_to_set, y_to_set, :] = b_coords

    return face_indices_map, b_coords_map


def my_color_map_by_proj(vertices_group, faces, renderer, cams, face_indices_map, b_coords_map, source_images=None,
                         segs=None,
                         save_path='texture.png', filter_by_seg=False):
    # due to multi-camera setup, we may have multiple group of
    # vertices and faces. as they all share the same face, so we
    # only pass vertices to list , while keeping the same face
    nCams = len(cams)
    texture_maps = []
    weights = []
    vis = [None] * nCams
    device = vertices_group[0].device
    faces = faces.detach().cpu().numpy()[0]

    for i in range(nCams):
        vertices = vertices_group[i].detach().cpu().numpy()[0]
        print("working on camera %d" % i)
        alignment = Mesh(v=vertices, f=faces)
        (points, normals) = uv_to_xyz_and_normals(alignment, face_indices_map, b_coords_map)

        # add artificious vertices and normals
        alignment.points = points
        alignment.v = np.vstack((alignment.v, points))
        alignment.vn = np.vstack((alignment.vn, normals))

        img = source_images[i]

        camera = cams[i]
        # TODO: implementation here may not be correct. need to double check here
        cam_v = np.array([0, 0, 0]).astype(np.float64).reshape((1, -1))
        cam_vis_ndot = np.array(visibility_compute(v=alignment.v, f=alignment.f, n=alignment.vn, \
                                                   cams=(cam_v)))
        # original implementation
        # cams[i].v = points
        # cam_vis_ndot = np.array(visibility_compute(v=alignment.v, f=alignment.f, n=alignment.vn, \
        #                                            cams=(np.array([camera.t.r.flatten()]))))

        cam_vis_ndot = cam_vis_ndot[:, 0, :]

        if filter_by_seg and segs is not None:
            # build a new function to extract cam_vis_ndot
            # seg mask : segs
            vertices_group = np.concatenate([vertices, points], axis=0)
            cam_vis_ndot = visibility_compute_with_seg(cam_vis_ndot, vertices_group, renderer, camera, segs, device)

        (cmap, vmap) = camera_projection(alignment, renderer, camera, cam_vis_ndot, img, face_indices_map, b_coords_map,
                                         device=device)

        texture_maps.append(cmap)
        weights.append(vmap)

        cv2.imwrite('texture_' + str(i) + '.png', texture_maps[i])
        cv2.imwrite('texture_w_' + str(i) + '.png', 255 * vmap)

        # restore old vertices and normals
        alignment.v = alignment.v[:(len(alignment.v) - points.shape[0])]
        alignment.vn = alignment.vn[:(len(alignment.vn) - points.shape[0])]
        del alignment.points

    # Create a global texture map
    # Federica Bogo's code
    sum_of_weights = np.array(weights).sum(axis=0)
    sum_of_weights[sum_of_weights == 0] = .00001
    for weight in weights:
        weight /= sum_of_weights

    if settings['max_tex_weight']:
        W = np.asarray(weights)
        M = np.max(W, axis=0)
        for i in range(len(weights)):
            B = weights[i] != 0
            weights[i] = (W[i, :, :] == M) * B

    # if clean_from_green:
    #     print('Cleaning green pixels')
    #     weights_green = clean_green(texture_maps, source_images, silhs)
    #     weights_all = [weights_green[i] * w for i, w in enumerate(weights)]
    # else:
    weights_all = weights

    sum_of_weights = np.array(weights_all).sum(axis=0)
    if not settings['max_tex_weight']:
        sum_of_weights[sum_of_weights == 0] = .00001
        for w in weights_all:
            w /= sum_of_weights

    full_texture_med = np.median(np.array([texture_maps]), axis=1).squeeze() / 255.
    T = np.array([texture_maps]).squeeze() / 255.
    W = np.zeros_like(T)
    if nCams > 1:
        W[:, :, :, 0] = np.array([weights_all]).squeeze()
        W[:, :, :, 1] = np.array([weights_all]).squeeze()
        W[:, :, :, 2] = np.array([weights_all]).squeeze()
    else:
        W[:, :, 0] = np.array([weights_all]).squeeze()
        W[:, :, 1] = np.array([weights_all]).squeeze()
        W[:, :, 2] = np.array([weights_all]).squeeze()

    # Average texture
    for i, texture in enumerate(texture_maps):
        for j in range(texture.shape[2]):
            texture[:, :, j] = weights_all[i] * texture[:, :, j] / 255.
    full_texture = np.sum(np.array([texture_maps]), axis=1).squeeze()
    cv2.imwrite(save_path, full_texture * 255)
    return full_texture, sum_of_weights


def camera_projection(alignment, renderer, camera, cam_vis_ndot, image, face_indices_map, b_coords_map,
                      dist=None, masked=False, device='cpu'):
    if not hasattr(alignment, 'points'):
        raise AttributeError('Mesh does not have uv_to_xyz points...')
    # cmap vmap -> color_map vis_map
    vis = cam_vis_ndot[0][-alignment.points.shape[0]:]
    n_dot = cam_vis_ndot[1][-alignment.points.shape[0]:]
    vis[n_dot < 0] = 0
    n_dot[n_dot < 0] = 0

    cmap = np.zeros((face_indices_map.shape[0], face_indices_map.shape[1], 3)) if not masked else \
        np.ones((face_indices_map.shape[0], face_indices_map.shape[1], 3)) * -1
    vmap = np.zeros(face_indices_map.shape)

    if len(alignment.points[vis == 1])>0:
        # set camera radial distortion parameters equal to 0, since we are already working on undistorted images
        # (tmp_proj,J) = cv2.projectPoints(alignment.points[vis==1], camera.r, camera.t, camera.camera_matrix, distCoeffs=np.zeros(5))
        # tmp_proj = camera.r[vis == 1]
        # im_coords = np.fliplr(np.atleast_2d(np.around(tmp_proj.squeeze()))).astype(np.int32)

        # here we use nmr implementation to find correct points to cast from 3d to 2d
        candidate_points = alignment.points[vis == 1]
        candidate_points = torch.tensor(candidate_points).unsqueeze(0).to(device)
        camera_gpu = camera.clone().to(device)
        im_coords = renderer.project_points(candidate_points.double(), camera_gpu.double())
        im_coords = im_coords.detach().cpu()[0].numpy().astype(np.int32)


        part_face_indices_map = np.copy(face_indices_map)
        not_vis_pixels = np.zeros(len(alignment.points))
        not_vis_pixels[vis == 0] = -1
        part_face_indices_map[face_indices_map != -1] = not_vis_pixels
        pixels_to_set = np.array(np.where(part_face_indices_map != -1)).T

        # this check might be unnecessary
        inside_image = np.where(np.logical_and(im_coords[:, 0] < np.shape(image)[0], im_coords[:, 0] >= 0))[0]
        inside_image = np.intersect1d(inside_image, np.where(
            np.logical_and(im_coords[:, 1] < np.shape(image)[1], im_coords[:, 1] >= 0))[0])
        pixels_to_set = pixels_to_set[inside_image]
        im_coords = im_coords[inside_image]
        # end check

        cmap[pixels_to_set[:, 0], pixels_to_set[:, 1], :] = image[im_coords[:, 0], im_coords[:, 1], :]
        vmap[pixels_to_set[:, 0], pixels_to_set[:, 1]] = n_dot[vis == 1][inside_image]

    return cmap, vmap  # , dmap)

def visibility_compute_with_seg(vis_ndot, points, renderer, camera, seg_mask, device):
    """
    only available with seg_mask
    1. call vis check, kick out non-visible vertexs (done in previous function)
    2. reproject all vertexs to image
    3. run check on vis_ndot matrix, if points are not foreground, kick out (reset vis and corresponding ndot)
    4. return updated ndot_vis
    : param point: dummy points to project, available upon call
    : param renderer: NMR (or whatever it works)
    : param seg_mask: gt seg_mask, filter out pixels not in valid range
    """

    # step 1: projection
    # assuming the length of the vis == length of points
    if isinstance(seg_mask, list):
        seg_mask = seg_mask[0]
    vis_matrix = vis_ndot[0,:] # 0 not visible 1:visible
    ndot_matrix = vis_matrix[1,:]
    candidate_points = torch.tensor(points).unsqueeze(0).to(device)
    camera_gpu = camera.clone().to(device)
    im_coords = renderer.project_points(candidate_points.double(), camera_gpu.double())
    im_coords = im_coords.detach().cpu()[0].numpy().astype(np.int32)

    # step 2: valid useful coords; update in sequence
    for i in range(points.shape[0]):
        y, x = im_coords[i].tolist()
        if seg_mask[y, x] == 0 and vis_matrix[i]>0:
            vis_matrix[i] = ndot_matrix[i] = 0

    vis_matrix = np.array(vis_matrix).reshape((1, -1))
    ndot_matrix = ndot_matrix.reshape((1, -1))
    vis_ndot = np.concatenate([vis_matrix, ndot_matrix], 0)
    return vis_ndot

def export_obj(path_prefix, vertex_positions, texture, faces, uvs, face_textures, is_point_cloud=False):
    """
    format for all required conponents
    v_p, faces, uvs, face_textures need list or numpy?
    vertex_positions : torch.tensor [962, 3]
    texture : torch.tensor [3, 512, 512]
    uv : torch.tensor [1054, 2]
    face_texture : torch.tensor [1920, 3]
    faces : torch.tensor [1920, 3]
    unwrap all tensors accordingly
    """

    # if the range of the texture is [-1, 1] : transfer to [0, 1], directly operates on matrix
    assert len(vertex_positions.shape) in [2, 3]
    mesh_path = path_prefix + '.obj'
    material_path = path_prefix + ".mtl"
    material_name = os.path.basename(path_prefix)

    if len(vertex_positions.shape) == 2:
        # shape : [B, num_verts, 3]
        vertex_positions = vertex_positions.unsqueeze(0)

    # export mesh .obj
    with open(mesh_path, 'w') as file:
        f_offset, ft_offset = 0, 0
        for mesh_id in range(vertex_positions.shape[0]):
            print(f'o mesh{mesh_id}', file=file)
            if texture is not None:
                print('mtllib ' + os.path.basename(material_path), file=file)
            for v in vertex_positions[mesh_id]:
                print('v {:.5f} {:.5f} {:.5f}'.format(*v), file=file)
            for uv in uvs:
                print('vt {:.5f} {:.5f}'.format(*uv), file=file)
            if texture is not None:
                print('usemtl ' + material_name, file=file)
            if not is_point_cloud:
                for f, ft in zip(faces, face_textures):
                    f = f + f_offset
                    ft = ft + ft_offset
                    print('f {}/{} {}/{} {}/{}'.format(f[0]+1, ft[0]+1, f[1]+1, ft[1]+1, f[2]+1, ft[2]+1), file=file)
                f_offset += vertex_positions.shape[1]
                ft_offset += uvs.shape[0]
    if texture is not None:
        # export material .mtl
        with open(material_path, 'w') as file:
            print('newmtl ' + material_name, file=file)
            print('Ka 1.000 1.000 1.000', file=file)
            print('Kd 1.000 1.000 1.000', file=file)
            print('Ks 1.000 1.000 1.000', file=file)
            print('d 1.0', file=file)
            print('illum 1', file=file)
            print('map_Ka '+ material_name + '.png', file=file)
            print('map_Kd '+ material_name + '.png', file=file)
    return