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


def imshow(tensor, title=None):
        # inverse-normalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor = tensor * std[:, None, None] + mean[:, None, None]
        
        # turn imgs into numpy array
        np_image = tensor.numpy().transpose((1, 2, 0))
        
        # constrain to [0,1]
        np_image = np.clip(np_image, 0, 1)
        
        # show the images
        plt.imshow(np_image)
        if title:
            plt.title(title)
        plt.axis('off')
    
        plt.show()


def get_dinov2_feature(imgs_path:str, img_size:list ,save_feature:bool=False, feature_save_path:str=None):

    print('=' * 25 + 'DINOv2 Features Getting' + '=' * 25)

    # check path to input imgs
    if not os.path.exists(imgs_path):
        raise SystemExit("imgs_path does not exist")


    if imgs_path.endswith('/'):
        img_dir = imgs_path
    else:
        img_dir = imgs_path + '/'

    # DINOv2 load from torch.hub
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    
    # Load the largest dino model (git clone)
    #dino = hubconf.dinov2_vitg14( )
    dinov2_vitg14 = dinov2_vitg14.cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load imgs to RGB format
    image_files = os.listdir(img_dir)
    images = []
    for image in image_files:
        img = Image.open(os.path.join(img_dir, image)).convert('RGB')
        images.append(img)
        

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

    # get batch size
    batch_size = len(images) 
    imgs_tensor = torch.zeros(batch_size, 3, patchs_h * 14, patchs_w * 14)
    
    # preprocess imgs and turn to tensor 
    for i, img in enumerate(images[:batch_size]):
        imgs_tensor[i] = transform(img)[:3]
    
    # load imgs to CUDA
    imgs_tensor = imgs_tensor.to(device)
    # get DINOv2 feature
    with torch.no_grad():
        features_dict = dinov2_vitg14.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']

    

    # Compute PCA between the patches of the image
    features = features.reshape(batch_size, patchs_h, patchs_w, 1536)

    print('=' * 25 + 'DINOv2 Features Got' + '=' * 25)

    if save_feature == True:
        print('=' * 25 + 'DINOv2 Features Saving' + '=' * 25)
        if os.path.exits(feature_save_path):
            i = 0
            for image_name in image_files:
                name_list = image_name.split('.')
                name_idex = name_list[0]
                save_path = os.path.join(feature_save_path, name_idex+".pt")
                torch.save(features[i], save_path)
        else:
            raise SystemExit("Feature Save Path Does Not Exit")


    return features

     


if __name__ == "__main__":
    
    parser = ArgumentParser(description='Get DINOv2 Feature Script Parameters')

    parser.add_argument("-p", "--imgs_path", type=str, default=None, help="path to the imgs directory")
    parser.add_argument("-s", "--imgs_size", type=list, default=[1264, 832], help="the shape of input imgs in format [hight, width]")

    args = parser.parse_args(sys.argv[1:])

    [img_h, img_w] = args.imgs_size
    patchs_h = int(img_h / 14)
    patchs_w = int(img_w / 14)

    features = get_dinov2_feature(args.imgs_path, args.imgs_size)
    batch_size = features.shape[0]

    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    pca_features = pca_features.reshape((batch_size, patchs_h * patchs_w, 3))
    
    # Visualize the first PCA component
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(pca_features[i : (i+1) , :, 0].reshape(patchs_h, patchs_w))
    plt.savefig("feature.jpg")
    plt.show()