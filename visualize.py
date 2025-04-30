import os.path

import numpy as np
import torch
import cv2
import importlib
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tool import imutils
import voc12.data
from torch.utils.data import DataLoader
import argparse
import torchvision


def visualize(normalized_heatmap, original=None):
    map_img = np.uint8(normalized_heatmap * 255)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
    if original is not None:
        original_img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        img = cv2.addWeighted(heatmap_img, 0.6, original_img, 0.4, 0)
    else:
        img = heatmap_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def blend_im(im, seg, alpha=0.7):
    pil_im = Image.fromarray(np.uint8(im))
    # pil_im = pil_im.transpose(Image.FLIP_LEFT_RIGHT)
    pil_seg = Image.fromarray(np.uint8(seg))
    im_blend = Image.blend(pil_im, pil_seg, alpha).convert("RGB")
    return np.asarray(im_blend)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        np.asarray,
        imutils.Normalize(),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_list", default="./voc12/train.txt", type=str)
    parser.add_argument("--voc12_root", default='../VOCdevkit/VOC2012', type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--weights", default='agcam_7253.pth', type=str)

    args = parser.parse_args()

    model = getattr(importlib.import_module('network.conformer_CAM'), 'Net_sm')()
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'net' in checkpoint.keys():
        checkpoint = checkpoint['net']
    else:
        checkpoint = checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model.eval()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  inter_transform=torchvision.transforms.Compose(
                                                      [
                                                          np.asarray,
                                                          imutils.Normalize(),
                                                          imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        N, C, H, W = img_list[0].size()
        img_name = img_name[0]

        label_numpy = label[0].cpu().numpy()
        label = label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        img = transform(Image.open(img_path))

        output = model(img.unsqueeze(0).cuda())
        cams = output['trancam']
        cams = F.interpolate(cams, orig_img_size, mode='bilinear', align_corners=False).detach()
        cams = cams.cpu().numpy()[0][1:]
        cams[cams < 0] = 0
        cam_max = np.max(cams, (1, 2), keepdims=True)
        cam_min = np.min(cams, (1, 2), keepdims=True)
        norm_cam = (cams - cam_min) / (cam_max - cam_min + 1e-5)

        indices = np.where(label_numpy == 1)[0]
        num = len(indices)
        print(iter)
        name = img_name
        os.makedirs('visualize/', exist_ok=True)
        for i in indices:
            cam = norm_cam[i]
            result_img = visualize(cam, orig_img)
            heatmap_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('visualize/', str(name) + '_' + str(i) + '.jpg'), heatmap_img_bgr)
