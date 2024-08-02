from datasets.change_detection import ChangeDetection_SECOND
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map
from utils.metric import IOUandSek

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from PIL import Image
# import shutil
import torch
# import torchcontrib
from utils.loss import TverskyLoss, js_div, Similarity
from torch.nn import CrossEntropyLoss, BCELoss, DataParallel, NLLLoss
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.cuda.set_device(torch.device("cuda:0"))
from utils.loss import Poly1CrossEntropyLoss
CUDA_LAUNCH_BLOCKING=1
import csv


class Trainer:

    def __init__(self, args):
        self.args = args

        trainset = ChangeDetection_SECOND(root=args.data_root, mode="train")
        valset = ChangeDetection_SECOND(root=args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=8, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=8, drop_last=False)
        self.model = get_model(args.model, args.backbone, args.pretrained,
                               len(trainset.CLASSES) - 1, args.lightweight)
        if args.pretrain_from:
            self.model.load_state_dict(torch.load(args.pretrain_from), strict=False)

        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        weight = torch.FloatTensor([2, 1, 2, 2, 1, 1]).cuda()

        self.criterion = Poly1CrossEntropyLoss(num_classes=6, weight=weight, ignore_index=-1)

        self.criterion_similarity = Similarity()
        self.criterion_bin = BCELoss(reduction='none')
        self.criterion_bin_2 = TverskyLoss()

        self.optimizer = AdamW([{"params": [param for name, param in self.model.named_parameters()
                                            if "backbone" in name], "lr": args.lr},
                                {"params": [param for name, param in self.model.named_parameters()
                                            if "backbone" not in name], "lr": args.lr * 1}],
                               lr=args.lr, weight_decay=args.weight_decay)
        self.model = DataParallel(self.model).cuda()

        self.iters = 0
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0
        self.seg_best = 0.0
        self.change_best = 0.0

        self.results_file = os.path.join('results.csv')  # Initialize results file

        # Write headers to results file
        with open(self.results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'miou', 'sek', 'score', 'oa'])

    def log_results(self, epoch, miou, sek, score, oa):
        with open(self.results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, miou, sek, score, oa])

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        total_loss_sem = 0.0
        total_loss_bin = 0.0
        total_loss_similarity = 0.0

        for i, (img1, img2, mask1, mask2, mask_bin, id) in enumerate(tbar):
            img1, img2 = img1.cuda(), img2.cuda()
            mask1, mask2 = mask1.cuda(), mask2.cuda()
            mask_bin = mask_bin.cuda()
            out1, out2, out_bin = self.model(img1, img2)

            loss1 = self.criterion(out1, mask1 - 1)
            loss2 = self.criterion(out2, mask2 - 1)

            loss_similarity = self.criterion_similarity(out1, out2, mask_bin)

            loss_bin = self.criterion_bin(out_bin, mask_bin)
            loss_bin_2 = self.criterion_bin_2(out_bin, mask_bin)

            try:
                loss_bin[mask_bin == 1] *= 2  # 将变化区域的loss值乘2
            except RuntimeError as e:
                print(f"Runtime error: {e}")
                print(f"loss_bin: {loss_bin}")
                print(f"mask_bin: {mask_bin}")
                continue

            loss_bin = loss_bin.mean()

            loss = (loss_bin + loss_bin_2) * 2 + loss1 + loss2

            total_loss_sem += loss1.item() + loss2.item()
            total_loss_similarity += loss_similarity.item()
            total_loss_bin += loss_bin.item() + loss_bin_2.item()
            total_loss += loss.item()

            self.iters += 1
            if args.warmup:
                warmup_steps = len(self.trainloader) * 20
                if warmup_steps and self.iters < warmup_steps:
                    warmup_percent_done = self.iters / warmup_steps
                    lr = args.lr * warmup_percent_done
                else:
                    lr = self.args.lr * (1 - self.iters / self.total_iters) ** 0.9
            else:
                lr = self.args.lr * (1 - self.iters / self.total_iters) ** 0.9
                lr = max(lr, 0.000015)
            self.optimizer.param_groups[0]["lr"] = lr
            if args.pretrain_from:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0
            else:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f" %
                                 (total_loss / (i + 1), total_loss_sem / (i + 1), total_loss_bin / (i + 1),
                                  total_loss_similarity / (i + 1)))

    def validation(self, epoch):  # Add epoch as a parameter
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric = IOUandSek(num_classes=len(ChangeDetection_SECOND.CLASSES))
        if self.args.save_mask:
            cmap = color_map()

        with torch.no_grad():
            for img1, img2, mask1, mask2, mask_bin, _ in tbar:
                img1, img2 = img1.cuda(), img2.cuda()

                out1, out2, out_bin = self.model(img1, img2, self.args.tta)
                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1
                out_bin = (out_bin > 0.5).cpu().numpy().astype(np.uint8)
                out1[out_bin == 0] = 0
                out2[out_bin == 0] = 0

                if self.args.save_mask:
                    for i in range(out1.shape[0]):
                        mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        mask.save("outdir/masks/val/im1/" + id[i])

                        mask = Image.fromarray(out2[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        mask.save("outdir/masks/val/im2/" + id[i])

                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out2, mask2.numpy())
                score, miou, sek, oa = metric.evaluate_SECOND()

                tbar.set_description("miou: %.4f, sek: %.4f, score: %.4f, oa:%.4f" % (miou, sek, score, oa))
        if self.args.load_from:
            exit(0)

        if score >= self.previous_best:
            if self.previous_best != 0:
                model_path = "outdir/%s_%s_%.4f.pth" % \
                             (self.args.model, self.args.backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)
            torch.save(self.model.module.state_dict(), "outdir/%s_%s_%.4f.pth" %
                       (self.args.model, self.args.backbone, score))
            self.previous_best = score

        self.log_results(epoch, miou, sek, score, oa)  # Log the results after validation


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    if args.load_from:
        trainer.validation(0)  # Pass 0 as epoch for initial validation

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.3f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training()
        trainer.validation(epoch)
