# python inference.py -- backbone 'resnet50' --model 'deeplabv3plus' --load-from 'path'
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image

from datasets.change_detection import ChangeDetection_SECOND
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map
from utils.metric import IOUandSek

import shutil
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, BCELoss, DataParallel
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import DataLoader
if __name__ == '__main__':
    # Your code here

    args = Options().parse()
    valset = ChangeDetection_SECOND(root=args.data_root, mode="val")
    valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                pin_memory=True, num_workers=8, drop_last=False)

    model = get_model(args.model, args.backbone, args.pretrained,
                      len(ChangeDetection_SECOND.CLASSES)-1, args.lightweight)
    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=False)

    model = DataParallel(model).cuda()
    tbar = tqdm(valloader)
    model.eval()
    metric = IOUandSek(num_classes=len(ChangeDetection_SECOND.CLASSES))
    with torch.no_grad():
        for img1, img2, mask1, mask2, _, id in tbar:

            img1, img2 = img1.cuda(), img2.cuda()

            out1, out2, out_bin = model(img1, img2, args.tta)
            out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
            out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1
            out_bin = ((out_bin > 0.5).cpu().numpy()).astype(np.uint8)

            out1[out_bin == 0] = 0
            out2[out_bin == 0] = 0
            args.save_mask = True
            cmap = color_map()
            # if args.save_mask:
            #     for i in range(out1.shape[0]):
            #         mask = Image.fromarray(out1[i].astype(np.uint8)).convert('P')
            #         mask.putpalette(cmap)
            #
            #         mask.save("outdir/mask2/second_val/out1/" + id[i])
            #
            #         mask = Image.fromarray(out2[i].astype(np.uint8)).convert('P')
            #         mask.putpalette(cmap)
            #         mask.save("outdir/mask2/second_val/out2/" + id[i])

            metric.add_batch(out1, mask1.numpy())
            metric.add_batch(out2, mask2.numpy())

        metric.color_map_SECOND()

        score, miou, sek, oa, iou2, F1, kappa, pr, re = metric.evaluate_inference()

        print('==>score', score)
        print('==>miou', miou)
        print('==>sek', sek)
        print('==>oa', oa)
        print('==>iou2', iou2)
        print('==>F1', F1)
        print('==>kappa', kappa)
        print('==>pr', pr)
        print('==>re', re)