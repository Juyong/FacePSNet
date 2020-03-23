
import torch
import numpy as np
import dlib
import cv2

def crop_cvimg(img, center_x, center_y, crop_width, crop_height, w_size, h_size):
    y1 = center_y - int(crop_height/2)
    y2 = y1 + crop_height
    x1 = center_x - int(crop_width/2)
    x2 = x1 + crop_width
    if x1 < 0:
        x2 = crop_width
        x1 = 0
    if y1 < 0:
        y2 = crop_height
        y1 = 0
    if x2 > img.shape[1]-1:
        x1 = img.shape[1]-1-crop_width
        x2 = img.shape[1]-1
    if y2 > img.shape[0]-1:
        y1 = img.shape[0]-1-crop_height
        y2 = img.shape[0]-1
    img_crop = img[y1:y2, x1:x2]
    img_crop = cv2.resize(img_crop, (w_size, h_size),
                          interpolation=cv2.INTER_NEAREST)
    return img_crop


def process_imgs(list_path):
    with open(list_path) as f:
        img_paths = f.read().splitlines()
    
    proxyimg_path = img_paths[0]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

    cur_id = 1
    proxyimg = dlib.load_rgb_image(proxyimg_path)
    dets, scores, idx = detector.run(proxyimg)
    score = -1
    best_id = -1
    for i in range(len(dets)):
        if scores[i]>score:
            score = scores[i]
            best_id = i
    if best_id == -1:
        return {'success': 0}
    shape = predictor(proxyimg, dets[best_id])
    lands = np.zeros((68,2), np.float32)
    for i in range(68):
        lands[i,0] = shape.part(i).x
        lands[i,1] = shape.part(i).y
    lms_x = lands[: ,0]
    lms_y = lands[: ,1]
    x_min = np.amin(lms_x)
    x_max = np.amax(lms_x)
    x_center = (x_max+x_min)/2
    y_top = 2*lms_y[19]-lms_y[29]
    y_bot = lms_y[8]
    y_len = y_bot-y_top
    y_top = y_top-0.1*y_len
    y_center = (y_top+y_bot)/2
    crop_width = (x_max-x_min)*1.1
    crop_height = (y_bot - y_center) * 2 * 1.1
    crop_width_34 = max(crop_width, crop_height * 3 / 4)

    center_x = int(x_center)
    center_y = int(y_center)
    crop_width = int(crop_width)
    crop_width = crop_width + (-crop_width) % 3
    crop_height = int(crop_width/3*4)
    
    w_size_proxy = int(150)
    h_size_proxy = int(200)
    w_size = int(600)
    h_size = int(800)

    cam = torch.tensor(
        (535*4, w_size/2, h_size/2), dtype=torch.float)
    
    imgs = []
    for i in range(len(img_paths)):
        cvimg = cv2.imread(img_paths[i], cv2.IMREAD_UNCHANGED)
        img = crop_cvimg(cvimg, center_x, center_y, crop_width,
                            crop_height, w_size, h_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        if i == 0:
            img_proxy = crop_cvimg(cvimg, center_x, center_y, crop_width, crop_height,
                                    w_size_proxy, h_size_proxy)
            img_proxy = cv2.cvtColor(img_proxy, cv2.COLOR_BGR2RGB)
            img_proxy = np.transpose(img_proxy, (2, 0, 1))
        imgs.append(img)

    img = np.concatenate(imgs, 0)
    img_tensor = torch.as_tensor(img).view(-1, 3, h_size, w_size)
    img_proxy_tensor = torch.as_tensor(img_proxy).unsqueeze(0)
    cam = cam.unsqueeze(0)
    return {'success': 1, 'img': img_tensor, 'img_proxy': img_proxy_tensor, 'cam': cam}
