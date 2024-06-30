from tqdm import tqdm

import os
from models import dptv2_vits, dptv2_vitb, dptv2_vitl, dptv2_vitg
import argparse
import numpy as np

from datasets import VOCSegmentation, Cityscapes

import torch

from PIL import Image
from pathlib import Path
from glob import glob
from timm.models import create_model


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str,
                        default='/mnt/d/CityScapesDataset/leftImg8bit/val/frankfurt/',
                        help="path to a single image or image directory")
    parser.add_argument("--nb_classes", type=int, default=19, choices=[19, 21],
                        help='19 for cityscapes, 21 for voc datasets')
    parser.add_argument("--datasets", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')


    parser.add_argument("--model", type=str, default='dptv2_vits',
                        choices=['dptv2_vits', 'dptv2_vitb', 'dptv2_vitl', 'dptv2_vitg'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default='./save_pred',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--ckpt", default='./output/dptv2_vits_best_model.pth', type=str,
                        help="resume from checkpoint")

    return parser


def main(opts):
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)


    model = create_model(
        opts.model,
        num_classes = opts.nb_classes,
        args=opts
    )

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model.to(device)

    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)

            pred = model.infer_image(img).astype(int)
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))


if __name__ == '__main__':
    opts = get_argparser().parse_args()
    if opts.save_val_results_to:
        Path(opts.save_val_results_to).mkdir(parents=True, exist_ok=True)
    main(opts)