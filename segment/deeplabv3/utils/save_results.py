import torch
import numpy as np
from PIL import Image
import os

def save_image(output_image, image_path):
    # if not isinstance(output_image, np.ndarray):
    #     if isinstance(output_image, torch.Tensor):  # get the data from a variable
    #         image_tensor = output_image.data
    #     # else:
    #     #     return output_image
    #     image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    #     if image_numpy.shape[0] == 1:  # grayscale to RGB
    #         image_numpy = np.tile(image_numpy, (3, 1, 1))
    #     # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    # else:  # if it is a numpy array, do nothing
    #     image_numpy = output_image
    # output_image = output_image.data.cpu()
    if output_image.shape[0] == 3:
        image_numpy = output_image.permute(1, 2, 0).cpu().numpy()
        image_numpy = (image_numpy * 255).astype('uint8')
    else:
        image_numpy = output_image.cpu().numpy()
        image_numpy = (image_numpy * 255/2).astype('uint8')
    # image_numpy = (image_numpy * 255).astype('uint8')
    image_pil = Image.fromarray(image_numpy)
    # output_image = output_image.data.cpu()
    # img3 = Image.fromarray(t3.numpy())
    # image_pil = Image.fromarray(image_numpy)
    # if image_pil.mode != 'RGB':
    #     image_pil = image_pil.convert('RGB')
    # print(image_path)
    image_pil.save(image_path)

def save_image_tuples(images, labels, predictions, save_path):
    # print(images, labels, predictions)
    # print(images.shape, labels.shape, predictions.shape)
    # print(type(images), type(labels), type(predictions))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for idx in range(len(images)):
        # print(images[idx], labels[idx], predictions[idx])
        # print(images[idx].shape, labels[idx].shape, predictions[idx].shape)
        # print(type(images[idx]), type(labels[idx]), type(predictions[idx]))
        save_image(images[idx], save_path + f'skyimage{idx}.png')
        save_image(labels[idx], save_path + f'real_annotation{idx}.png')
        save_image(predictions[idx], save_path + f'predicted_annotation{idx}.png')
