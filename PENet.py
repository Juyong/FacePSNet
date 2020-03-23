import torch
import torch.nn as nn
import numpy
import gen_normal_mask
import torchvision

class ProxyEstimationNet(nn.Module):
    def __init__(self, path_3dmm, path_tris):
        super(ProxyEstimationNet, self).__init__()

        id_dim = 100
        exp_dim = 79
        point_num = 34920
        mu,b,sig_id, sig_exp = load_3dmm_file(path_3dmm, point_num, id_dim, exp_dim)

        self.mu = torch.as_tensor(mu).cuda()
        self.b = torch.as_tensor(b).cuda()
        tris = numpy.loadtxt(path_tris, int) - 1
        self.tris = torch.as_tensor(tris).cuda()
        self.height = 800
        self.width = 600
        self.sig_id = torch.as_tensor(sig_id).cuda()
        self.sig_exp = torch.as_tensor(sig_exp).cuda()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 500)

        self.reg_id = nn.Linear(500, id_dim)
        self.reg_exp = nn.Linear(500, exp_dim)
        self.reg_euler = nn.Linear(500, 3)
        self.reg_trans = nn.Linear(500, 3)
        self.center = torch.nn.Parameter(torch.tensor(
            (0, 0, -600), dtype=torch.float), False)

    def get_geo(self, pca_para):
        return torch.mm(pca_para, self.b) + self.mu

    def forward(self, input_rgb, cam):
        active_opt = nn.ReLU(True)
        fc2 = active_opt(self.resnet(input_rgb))
        norm_id_para = self.reg_id(fc2)
        id_para = norm_id_para*self.sig_id
        norm_exp_para = self.reg_exp(fc2)
        exp_para = norm_exp_para*self.sig_exp
        euler_angle = self.reg_euler(fc2)
        trans = self.reg_trans(fc2) + self.center
        pcapara_pred = torch.cat((id_para, exp_para), dim=1)
        geometry = self.get_geo(pcapara_pred).reshape(
            id_para.shape[0], -1, 3).permute(0, 2, 1)
        rott_geo, proj_geo = proj_rott_geo(geometry, euler_angle, trans, cam)
        tri_normal = compute_tri_normal(rott_geo, self.tris)
        normal_map, mask = generate_normal_mask(
            proj_geo.contiguous(), tri_normal.contiguous(), self.tris.contiguous(), self.height, self.width)
        pca_pose_cam = torch.cat(
                (id_para, exp_para, euler_angle, trans, cam), 1)
        return pca_pose_cam, normal_map, mask

def load_3dmm_file(path_3dmm, point_num, id_dim, exp_dim):
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
    return mu, b, sig_shape, sig_exp

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
    return rott_geo, proj_geo

def compute_tri_normal(geometry, tris):
    tri_1 = tris[:, 0]
    tri_2 = tris[:, 1]
    tri_3 = tris[:, 2]
    vert_1 = torch.index_select(geometry, 2, tri_1)
    vert_2 = torch.index_select(geometry, 2, tri_2)
    vert_3 = torch.index_select(geometry, 2, tri_3)
    nnorm = torch.cross(vert_2-vert_1, vert_3-vert_1, 1)
    normal = nn.functional.normalize(nnorm).permute(0, 2, 1)
    return normal

def generate_normal_mask(proj_geo, tri_normal, tri_inds, height, width):
    normak_map, mask = gen_normal_mask.gen_normal_mask(
        proj_geo, tri_normal, tri_inds, height, width)
    return normak_map, mask
