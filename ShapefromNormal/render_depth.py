import torch
import os
import numpy
import cv2
import sys
import gen_normal_mask

point_num = 34920
id_dim = 100
exp_dim = 79


def parase_3dmm_file(path_3dmm):
    fid = open(path_3dmm, 'rb')
    mu_shape = numpy.fromfile(fid, dtype='float32', count=3*point_num)
    b_shape = numpy.fromfile(fid, dtype='float32', count=3 *
                             point_num*id_dim).reshape(3*point_num, id_dim)
    sig_shape = numpy.fromfile(fid, dtype='float32', count=id_dim)
    mu_exp = numpy.fromfile(fid, dtype='float32', count=3*point_num)
    b_exp = numpy.fromfile(
        fid, dtype='float32', count=3*point_num*exp_dim).reshape(3*point_num, exp_dim)
    sig_exp = numpy.fromfile(fid, dtype='float32', count=exp_dim)

    b = numpy.transpose(numpy.concatenate((b_shape, b_exp), 1), (1, 0))
    mu = (mu_shape+mu_exp) / 1000.0
    sig_shape /= 1000.0
    sig_exp /= 1000.0

    fid.close()
    return mu, b


DEVICE = int(sys.argv[1])
torch.cuda.set_device(DEVICE)

mu, b = parase_3dmm_file('../ShapefromNormal/data/part_3dmm.bin')
mu = torch.as_tensor(mu).cuda()
b = torch.as_tensor(b).cuda()


def get_geo(pca_para):
    return torch.mm(pca_para, b) + mu


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = -euler_angle[:, 0].reshape(-1, 1, 1)
    phi = -euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def project_geo(rott_geo, camera_para):
    fx = camera_para[:, 0]
    fy = camera_para[:, 0]
    cx = camera_para[:, 1]
    cy = camera_para[:, 2]

    X = rott_geo[:, 0, :]
    Y = rott_geo[:, 1, :]
    Z = rott_geo[:, 2, :]

    fxX = fx[:, None]*X
    fyY = fy[:, None]*Y

    proj_x = -fxX/Z + cx[:, None]
    proj_y = fyY/Z + cy[:, None]

    return torch.cat((proj_x[:, :, None], proj_y[:, :, None], Z[:, :, None]), 2)


def proj_rott_geo(geometry, euler, trans, cam):
    rot = euler2rot(euler)
    rott_geo = torch.bmm(rot, geometry) + trans[:, :, None]
    proj_geo = project_geo(rott_geo, cam)
    return proj_geo


def generate_depth(proj_geo, tri_inds, height, width):
    depth = gen_normal_mask.gen_depth(
        proj_geo, tri_inds, height, width)[0]
    return depth


tris = numpy.loadtxt('../ShapefromNormal/data/tris.txt', int) - 1
tris = torch.as_tensor(tris).cuda()

with open(sys.argv[2]) as f:
    lines = f.read().splitlines()

for path in lines:
    os.chdir(path)
    pca_pose_cam = torch.as_tensor(numpy.loadtxt(
        'pca_pose_cam.txt', numpy.float32)).cuda()
    pca = pca_pose_cam[0:179].reshape(1, -1)
    euler = pca_pose_cam[179:182].reshape(1, -1)
    trans = pca_pose_cam[182:185].reshape(1, -1)
    cam = pca_pose_cam[185:188].reshape(1, -1)

    geo = get_geo(pca).reshape(pca.shape[0], -1, 3).permute(0, 2, 1)
    proj_geo = proj_rott_geo(geo, euler, trans, cam)
    height = 800
    width = 600
    depth = generate_depth(proj_geo.contiguous(), tris, height, width)
    # print(depth[0, ...].max())
    # print(depth[0, ...].min())
    depth_np = depth[0, ...].cpu().detach().numpy()
    depth_np.tofile('init_depth.bin')
    print(path, 'render_depth done')
