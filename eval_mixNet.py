import os
import time
import cv2
import numpy as np
import json
from shapely.geometry import *
import torch
import subprocess
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text, TD500HUSTText, ArtText, ArtTextJson, Mlt2019Text, Ctw1500Text_New, TotalText_New, Ctw1500Text_mid, TotalText_mid, TD500HUSTText_mid

from network.textnet import TextNet
from cfglib.config import config as cfg, update_config, print_config
from cfglib.option import BaseOptions
from util.augmentation import BaseTransform
from util.visualize import visualize_detection, visualize_gt
from util.misc import to_device, mkdirs,rescale_result, get_cosine_map
# from util.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_icdar15, \
#     deal_eval_TD500, data_transfer_ICDAR, data_transfer_TD500, data_transfer_TD500HUST, data_transfer_MLT2017

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def inference(model, test_loader, output_dir):

    total_time = 0.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        idx = 0  # test mode can only run with batch_size == 1
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        image = image.to(device,non_blocking=True)
        input_dict['img'] = image
        # get detection result
        start = time.time()
        with torch.no_grad():
            output_dict = model(input_dict)

        torch.cuda.synchronize()
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0

        print('detect {} / {} images: {}. ({:.2f} fps)'.format(i + 1, len(test_loader), meta['image_id'][idx], fps), end = '\r', flush = True)

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        if cfg.viz:
        # if True:
            gt_contour = []
            label_tag = meta['label_tag'][idx].int().cpu().numpy()
            for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
                if n_annot.item() > 0:
                    gt_contour.append(annot[:n_annot].int().cpu().numpy())

            gt_vis = visualize_gt(img_show, gt_contour, label_tag)
            show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)

            show_map = np.concatenate([heat_map, gt_vis], axis=1)
            show_map = cv2.resize(show_map, (320 * 3, 320))
            im_vis = np.concatenate([show_map, show_boundary], axis=0)

            path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0]+".jpg")
            cv2.imwrite(path, im_vis)
            # print(path)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)

        torch.cuda.empty_cache()

        # write to file

def main(vis_dir_path):

    cudnn.benchmark = False
    # Data
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'MixNet_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))

    model.load_model(model_path)
    model.to(cfg.device)  # copy to cuda
    model.eval()
    with torch.no_grad():
        print('Start testing MixNet.')
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        inference(model, test_loader, output_dir)
    print("{} eval finished.".format(cfg.exp_name))


