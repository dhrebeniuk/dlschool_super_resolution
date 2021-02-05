from PIL import Image
import numpy as np


def post_processed_image_from_torchtensor(tensor_image):
    images = tensor_image.cpu().numpy()
    for index in range(len(images)):
        img = np.transpose(images[index], (1, 2, 0))
        return img


def load_image(filename, width, height):
    img = Image.open(filename)
    if img.size[0] != width or img.size[1] != height:
        img = img.resize((width, height), Image.ANTIALIAS)
    return img


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
