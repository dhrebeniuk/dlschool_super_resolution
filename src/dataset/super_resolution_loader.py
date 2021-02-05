from torch.utils.data import Dataset
from ..utils.image_utils import *
import os


class SuperResolutionUrban100Dataset(Dataset):
    def __init__(self, folder_path, transforms=None):
        self.folder_path = folder_path
        self.transforms = transforms

        self.hi_scale_images = []
        self.low_scale_images = []

        files_list = os.listdir(folder_path)

        for file_name in files_list:
            if "_HR" not in file_name:
                continue

            file_path = os.path.join(folder_path, file_name)

            width = 1024
            height = 768

            hi_res_image = load_image(file_path, width // 4, height // 4)

            self.hi_scale_images.append(hi_res_image)

            low_scale_width = width // 8
            low_scale_height = height // 8

            low_res_image = load_image(file_path, low_scale_width, low_scale_height)

            self.low_scale_images.append(low_res_image)

    def __getitem__(self, idx):
        # load images and targets
        source = self.low_scale_images[idx]
        target = self.hi_scale_images[idx]

        if self.transforms is not None:
            source = self.transforms(source)

        if self.transforms is not None:
            target = self.transforms(target)

        return source, target

    def __len__(self):
        return len(self.low_scale_images)
