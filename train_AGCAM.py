import numpy as np
import torch
import os

from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils
import argparse
import importlib
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import random
import torchvision


def setup_seed(seed):
    print("random seed is set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--start_epoches", default=1, type=int)
    parser.add_argument("--max_epoches", default=30, type=int)
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--tf_freq", default=500, type=int, help="frequency")
    parser.add_argument("--val_freq", default=10, type=int)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--voc12_root", default='../VOCdevkit/VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/train_voc.txt", type=str)
    parser.add_argument("--arch", default='sm', type=str)
    parser.add_argument("--seed", default=15, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.session_name, exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'ckpt'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, args.session_name + '.log'))
    tblogger = SummaryWriter(os.path.join(args.session_name, 'runs'))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)()

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(320, 640),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                          hue=0.1),
                                                   np.asarray,
                                                   imutils.Normalize(),
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = voc12.data.VOC12ClsDatasetMSF(args.val_list, voc12_root=args.voc12_root,
                                                inter_transform=torchvision.transforms.Compose(
                                                    [
                                                        np.asarray,
                                                        imutils.Normalize(),
                                                        imutils.HWC_to_CHW]))
    val_data_loader = DataLoader(val_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_dec, eps=1e-8)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        else:
            checkpoint = checkpoint
        model_dict = model.state_dict()
        for k in ['trans_cls_head.weight', 'trans_cls_head.bias']:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
        for k in ['conv_cls_head.weight', 'conv_cls_head.bias']:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    bestiou = 0
    train_steps = 0
    print('loaded weight:', args.checkpoint)
    for epoch in range(args.max_epoches):
        print('Epoch %d/%d' % (epoch, args.max_epoches))
        for step, (_, img, valid_mask, label) in enumerate(train_data_loader):
            N, C, H, W = img.size()
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            valid_mask = valid_mask.cuda()
            valid_mask[:, 1:] = valid_mask[:, 1:] * label
            valid_mask_lowres = F.interpolate(valid_mask, size=(H // 16, W // 16), mode='nearest')

            outputs = model(img, valid_mask_lowres)

            score_cnn = outputs['score_cnn']
            score_trans = outputs['score_trans']
            norm_cam = outputs['cam']
            final_cam = outputs['final_cam']

            final_cam = final_cam / (F.adaptive_max_pool2d(final_cam, (1, 1)) + 1e-5)
            lossCLS = F.multilabel_soft_margin_loss((score_cnn + score_trans).unsqueeze(2).unsqueeze(3)[:, 1:, :, :],
                                                    label[:, 1:, :, :])
            lossAUX = torch.mean(torch.abs(final_cam - norm_cam))

            losses = lossCLS + lossAUX

            avg_meter.add({'lossCLS': lossCLS.item(), 'lossAUX': lossAUX.item()})

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            train_steps += 1

            if train_steps % args.print_freq == 0:
                timer.update_progress(train_steps / max_step)

                print('step:%5d/%5d' % (train_steps, max_step),
                      'lossCLS:%.4f' % (avg_meter.pop('lossCLS')),
                      'lossAUX:%.4f' % (avg_meter.pop('lossAUX')),
                      'imps:%.1f' % ((step + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.6f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_est_finish()), flush=True)

            else:
                timer.reset_stage()
