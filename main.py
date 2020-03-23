import torch
import os 
import numpy as np
import sys
import cv2
from PENet import ProxyEstimationNet
from NENet import NormalEstimationNet
from preprocess_data import process_imgs

# set gpu id
DEVICE = int(sys.argv[1]) 
torch.cuda.set_device(DEVICE)

# load proxy estimation network
model_ProxyEst = ProxyEstimationNet('data/part_3dmm.bin', 'data/tris.txt').cuda()
model_ProxyEst.load_state_dict(torch.load('trained_models/proxy.pth'), strict=False)

# load normal estimation netork
model_NormalEst = NormalEstimationNet().cuda()
model_NormalEst.load_state_dict(torch.load('trained_models/normal.pth.tar')['state_dict'])

model_ProxyEst.eval()
model_NormalEst.eval()

# input imgs list
imgs_list = sys.argv[2]

# output directory
output_dir = sys.argv[3]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = process_imgs(imgs_list)
if data['success'] == 0:
    print('no face detected')
else:
    img_proxy = data['img_proxy'].cuda().float()/255.0
    imgs = data['img'].cuda().float()/255.0
    cam = data['cam'].cuda()
    pca_pose_cam, normal_proxy, mask = model_ProxyEst(img_proxy, cam)
    normal_pred = model_NormalEst(imgs, normal_proxy)

    ## save files for geometry recovery
    normal_img = ((normal_pred[0, ...].data + 1) *
                30000*mask.data[0, ...]).permute(1, 2, 0)[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(output_dir, 'normal.png'),
                normal_img.cpu().detach().numpy().astype('uint16'))
    cv2.imwrite(os.path.join(output_dir, 'mask.jpg'),
                (mask[0, ...]*255).permute(1, 2, 0).cpu().detach().numpy().astype('uint8'))
    np.savetxt(os.path.join(output_dir, 'pca_pose_cam.txt'),
                pca_pose_cam[0, ...].detach().cpu().numpy(), '%f')

    