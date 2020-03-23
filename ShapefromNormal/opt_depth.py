import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy
import sys


def compute_tri_normal(geometry, tris):
    tri_1 = tris[:, 0]
    tri_2 = tris[:, 1]
    tri_3 = tris[:, 2]

    vert_1 = torch.index_select(geometry, 0, tri_1)
    vert_2 = torch.index_select(geometry, 0, tri_2)
    vert_3 = torch.index_select(geometry, 0, tri_3)

    nnorm = torch.cross(vert_2-vert_1, vert_3-vert_1, 1)

    return nnorm


def inverse_proj(depth, uv, camera_para):
    fx = camera_para[0]
    fy = camera_para[0]
    cx = camera_para[1]
    cy = camera_para[2]

    proj_x = uv[:, 0]
    proj_y = uv[:, 1]

    X = -(proj_x-cx)*depth/fx
    Y = (proj_y-cy)*depth/fy

    return torch.cat((X[:, None], Y[:, None], depth[:, None]), 1)


DEVICE = int(sys.argv[1])
torch.cuda.set_device(DEVICE)

learning_rate = float(sys.argv[2])
ITER_NUM = int(sys.argv[3])

with open(sys.argv[4]) as f:
    lines = f.read().splitlines()
for i in range(len(lines)):
    path = lines[i]
    print(path, 'opt_depth begin')
    os.chdir(path)
    uv = numpy.loadtxt('uvs.txt', numpy.int32)
    uv = torch.as_tensor(uv).cuda().float()

    normals = numpy.fromfile('normals.bin', numpy.float32)
    normals = torch.as_tensor(normals).cuda().reshape(-1, 3)

    normal_ids = numpy.fromfile('valid_normals.bin', numpy.int32)
    normal_ids = torch.as_tensor(normal_ids).cuda().long()

    lap_ids = numpy.fromfile('valid_laps.bin', numpy.int32)
    lap_ids = torch.as_tensor(lap_ids).cuda().long()

    for_normals = numpy.fromfile('for_normals.bin', numpy.int32)
    for_normals = torch.as_tensor(for_normals).cuda().long().reshape(-1, 3)

    for_laps = numpy.fromfile('for_laps.bin', numpy.int32)
    for_laps = torch.as_tensor(for_laps).cuda().long().reshape(-1)

    init_deps = numpy.fromfile('init_deps.bin', numpy.float32)
    init_deps = torch.as_tensor(init_deps).float().cuda()

    cam = numpy.loadtxt('pca_pose_cam.txt', numpy.float32)[-3:]
    cam = torch.as_tensor(cam).cuda()

    depth = torch.zeros_like(uv[:, 0], requires_grad=True)
    depth.data.fill_(0)

    optimizer = torch.optim.Adam([depth], lr=learning_rate)

    gt_normal = torch.index_select(normals, 0, normal_ids)

    L2Loss = torch.nn.MSELoss()
    L1Loss = torch.nn.L1Loss()

    for iter in range(ITER_NUM):
        depth_z = depth + init_deps
        geo = inverse_proj(depth_z, uv, cam)

        nnorm = compute_tri_normal(geo, for_normals).reshape(-1, 4, 3)
        nnorm = torch.mean(nnorm, dim=1)
        rec_normal = nn.functional.normalize(nnorm)

        geo_c = torch.index_select(geo, 0, lap_ids)
        geo_neighbor = torch.index_select(geo, 0, for_laps).reshape(-1, 4, 3)
        geo_neighbor = torch.mean(geo_neighbor, dim=1)
        loss_lap = L2Loss(geo_c[:, 2], geo_neighbor[:, 2])

        loss_bias = torch.mean(depth*depth)

        loss_normal = L2Loss(rec_normal, gt_normal)

        loss = loss_lap*12+loss_normal + loss_bias*0.0005
        if iter % 5000 == 4999 or iter == 0:
            print(iter+1, loss_normal.item(),
                  loss_bias.item(), loss_lap.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    depth_z = depth + init_deps
    numpy.savetxt('opt_z.txt', depth_z.cpu().detach().numpy(), '%f')
    print(path, 'opt_depth done')
