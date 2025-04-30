import numpy as np
import torch
import os
import voc12.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils
import argparse
from PIL import Image
import torch.nn.functional as F

from tool.imutils import crf_inference_label

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='agcam_7253.pth', type=str)
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='../VOCdevkit/VOC2012', type=str)
    parser.add_argument("--out_cam", default='data/cam', type=str)
    parser.add_argument("--out_crf", default=None, type=str)
    parser.add_argument("--arch", default='sm', type=str)
    # parser.add_argument("--method", default='transcam', type=str)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out_cam):
        os.makedirs(args.out_cam)

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

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  inter_transform=torchvision.transforms.Compose(
                                                      [
                                                          np.asarray,
                                                          imutils.Normalize(),
                                                          imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        N, C, H, W = img_list[0].size()
        img_name = img_name[0];
        label_val = label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
        label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]


        cam_list = []

        with torch.no_grad():
            for i, img in enumerate(img_list):
                outputs = model(img.cuda())
                cam = outputs['final_cam']

                cam = F.interpolate(cam[:, 1:, :, :], orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)
                cam_list.append(cam)

        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1, 2), keepdims=True)
        cam_min = np.min(sum_cam, (1, 2), keepdims=True)
        sum_cam[sum_cam < cam_min + 1e-5] = 0
        norm_cam = (sum_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        h, w = list(cam_dict.values())[0].shape
        tensor = np.zeros((21, h, w), np.float32)
        for key in cam_dict.keys():
            tensor[key + 1] = cam_dict[key]

        crf_alpha = [0.48]
        if args.out_crf is not None:
            for t in crf_alpha:
                tensor[0, :, :] = t
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
                crf = crf_inference_label(orig_img, predict)
                folder = args.out_crf + ('_%.2f'%t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                Image.fromarray(crf.astype(np.uint8)).save(os.path.join(folder, img_name + '.png'))

        if iter % 50 == 0:
            print("weights: {}".format(args.weights), "     iter: {}".format(iter))

