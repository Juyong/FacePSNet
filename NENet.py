import torch
import torch.nn as nn

class NormalEstimationNet(nn.Module):
    def __init__(self):
        super(NormalEstimationNet, self).__init__()
        self.extractor = ImgFeatExtractor()
        self.proxy_extractor = ProxyFeatExtractor()
        self.regressor = Normal_Regressor()

    def forward(self, imgs, proxy_normal):
        imgs_feat = self.extractor(imgs)
        img_feat_fused = imgs_feat.max(dim=0, keepdim=True)[0]
        proxy_feat = self.proxy_extractor(proxy_normal)
        feat_fused = torch.cat((img_feat_fused, proxy_feat), 1)
        normal = self.regressor(feat_fused, proxy_normal)
        return normal

class ImgFeatExtractor(nn.Module):
    def __init__(self):
        super(ImgFeatExtractor, self).__init__()
        self.conv1 = conv(
            3, 64,  k=3, stride=1, pad=1)
        self.conv2 = conv(
            64,   192, k=3, stride=2, pad=1)
        self.conv3 = conv(
            192,  192, k=3, stride=1, pad=1)
        self.conv4 = conv(
            192,  192, k=3, stride=2, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out_feat = self.conv4(out)
        return out_feat

class ProxyFeatExtractor(nn.Module):
    def __init__(self):
        super(ProxyFeatExtractor, self).__init__()
        self.conv1 = conv(
            3, 64,  k=3, stride=1, pad=1)
        self.conv2 = conv(
            64, 128, k=3, stride=2, pad=1)
        self.conv3 = conv(
            128, 128, k=3, stride=1, pad=1)
        self.conv4 = conv(
            128, 64, k=3, stride=2, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out_feat = self.conv4(out)
        return out_feat

class Normal_Regressor(nn.Module):
    def __init__(self):
        super(Normal_Regressor, self).__init__()
        self.conv5 = conv(
            256,  256, k=3, stride=1, pad=1)
        self.conv6 = deconv(256, 128)
        self.conv7 = conv(
            128, 128, k=3, stride=1, pad=1)
        self.deconv1 = conv(
            128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = conv(
            128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = deconv(128, 64)
        self.est_normal = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, normal_proxy):
        out = self.conv5(x)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        normal = self.est_normal(out) + normal_proxy
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

def conv(cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=k,
                    stride=stride, padding=pad, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )

def deconv(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, kernel_size=4,
                           stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )