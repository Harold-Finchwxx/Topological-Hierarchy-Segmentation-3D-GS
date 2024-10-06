import sys
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
#import hubconf ## Load the largest dino model (git clone)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from argparse import ArgumentParser

def images_convert(imgs_path:str, img_size:list, output_imgs_path:str):

    print("=" * 25 +'Images Convert Start' + '=' *25)

    # check path to input imgs
    if not os.path.exists(imgs_path):
        raise SystemExit("imgs_path does not exist")

    if imgs_path.endswith('/'):
        img_dir = imgs_path
    else:
        img_dir = imgs_path + '/'

    print('=' * 25 + "Input Path of Convert Checked" + "=" * 25)

    # check output directory
    if not os.path.exists(output_imgs_path):
        os.makedirs(output_imgs_path)

    print("=" * 25 + "Output Directories Established" + "=" * 25)

    [img_h, img_w] = img_size
    patchs_h = int(img_h / 14)
    patchs_w = int(img_w / 14)

    # img preprocess pipeline
    transform = T.Compose([
        T.Resize((patchs_h * 14, patchs_w * 14), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop((patchs_h * 14, patchs_w * 14)),
        T.ToTensor(),
        #T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # load imgs to RGB format
    image_files = os.listdir(img_dir)

    for image in image_files:
        img = Image.open(os.path.join(img_dir, image)).convert('RGB')
        outputfile = os.path.join(output_imgs_path, image)
        outimg = transform(img)
        outimg.save(outputfile)

    print("=" * 25 +'Images Convert Finished' + '=' *25)


if __name__ == "__main__":

    parser = ArgumentParser(description="Convert images into DINOv2 allowed shape")

    parser.add_argument("-i", "--images_path", type=str, default=None, help="path to the input images")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="path to the desired output directory")
    parser.add_argument("-s", "--image_size", type=list, default=[1264, 832], help="the desired shape of converted images")

    args = parser.parse_args(sys.argv[1:])

    images_convert(args.images_path, args.image_size, args.output_path)
        
    