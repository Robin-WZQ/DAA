import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import argparse
import cv2

def create_parser():
    parser = argparse.ArgumentParser(description='Analysis of Attention Maps')
    parser.add_argument('-np',
                        '--npy_path',
                        default='./attention_metrics_0.npy',
                        type=str,
                        required=True,
                        dest="npy_path")
    parser.add_argument('-o',
                        '--output_path',
                        default='./',
                        type=str,
                        required=False,
                        dest="output_path")
    
    args = parser.parse_args()
    return args

def main():
    args = create_parser()
    npy_path = args.npy_path
    output_path = args.output_path

    # load npy data
    data = np.load(npy_path)
    T, L, H, W = data.shape 

    # create output folder
    output_folder = 'dynamic_attention_maps'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # save each image as a PNG file
    for t in range(T):
        # create a single image from all attention maps
        img_data = data[t, :, :, :]
        img_data = torch.from_numpy(img_data)
        
        images = []
        for i in range(L):
            image = img_data[i, :, :]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            images.append(image)
        img_data = np.hstack(images)
        
        plt.imsave(f'{output_folder}/image_{t}.png', img_data)

if __name__ == '__main__':
    main()
