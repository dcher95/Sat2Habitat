import open_clip
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image, ImageFile
from datetime import datetime
from torchvision.transforms import v2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from config import config

class SatHabData(Dataset):
    def __init__(self, image_path, csv_path, mode='train', epoch=0, curriculum=5):
        self.image_path = Path(image_path)
        self.csv_path = csv_path
        self.image_dict = self._build_image_dict()
        self.mode = mode
        self.epoch = epoch
        self.curriculum = curriculum

        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.data = pd.read_csv(self.csv_path)
        self.data = self.data[self.data["key"].astype(str).isin(self.image_dict.keys())].reset_index(drop=True)
        
        # text params
        self.tokenizer = open_clip.get_tokenizer('hf-hub:MVRL/taxabind-vit-b-16')
        self.hab_desc = config.hab_desc
        self.alt_txt_cols = config.alt_txt_cols

        if mode == 'test':
            self.random_prob = 1
        else:
            self.random_prob = config.random_prob

        if mode == 'train':
            self.image_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.GaussianBlur(5, (0.01, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        else:
            self.image_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.data)
    
    def update_epoch(self, epoch):
        """Update the current epoch."""
        self.epoch = epoch
    
    def __getitem__(self, index):
        # Band-aid fix for truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        row = self.data.iloc[index]
        sat_id = row["key"].astype(int).astype(str)
        patch_key = row["patch"]

        lat = torch.tensor(row["lat"])
        lon = torch.tensor(row["lon"])

        # Get the text description (habitat or randomized)
        text = self._get_text_randomized(row)
        text_tokens = self.tokenizer(text)

        # Fetch the image path using the sat_id
        image_file = self.image_dict.get(sat_id)

        # Band-aid -- only when doing evaluations
        if self.mode == 'test':
            if image_file is None:
                # Log the error or skip this sample
                print(f"Warning: Image for sat_id {sat_id} not found.")
                return None, None, None  # Or handle it in some other way, like skipping this sample

        # Proceed to open the image and process it
        image = self.image_transform(Image.open(image_file))

        # If we are doing curriculum learning: image --> patch
        if self.epoch <= self.curriculum:
            return image, text_tokens, torch.tensor([lat, lon])
        
        else:
            # Stage 2: Patchify the image and use habitat descriptor and patch for contrastive learning
            patch = self._get_patch_for_image(image, patch_key)
            return patch, text_tokens, torch.tensor([lat, lon])

    
    def _build_image_dict(self):
        image_dict = {}
        for image_file in self.image_path.glob("*.jpg"):
            try:
                sat_id = image_file.stem.split("/")[-1].replace(".jpg" , "")
                image_dict[sat_id] = image_file
            except ValueError:
                print(f"Invalid image file name {image_file}")
        return image_dict
    
    def _get_text_randomized(self, row):

        if np.random.rand() < self.random_prob:
            return row[self.hab_desc]
        
        else:    
            alternative_values = row[self.alt_txt_cols].to_numpy()
            non_nan_values = alternative_values[pd.notna(alternative_values)]
            
            # If there are non-NaN values, select one randomly
            if non_nan_values.size > 0:
                return np.random.choice(non_nan_values)
            
            # If all alternatives are NaN, return 'habitat' as a fallback
            return row[self.hab_desc]
        
    def _get_patch_for_image(self, image, patch_key, num_patches=3):
        """
        This function will take the image and the patch_key (in the format 'patch_row_patch_col'),
        and return the patch corresponding to the specified row and column.
        """
        # Parse the patch_key to get the row and column indices
        patch_row, patch_col = map(int, patch_key.split('_'))
        
        # Image size
        image_width, image_height = image.size
        
        # Define how many patches you want to divide the image into (e.g., 3x3 grid)
        patch_width = image_width // num_patches
        patch_height = image_height // num_patches
        
        # Calculate the coordinates of the patch based on patch_row and patch_col
        left = patch_col * patch_width
        upper = patch_row * patch_height
        right = left + patch_width
        lower = upper + patch_height
        
        # Crop the patch from the image
        patch = image.crop((left, upper, right, lower))
        
        # Return the transformed patch
        return self.image_transform(patch)

        
if __name__ == '__main__':
    im_dir = "/data/cher/Sat2Habitat/data/bing_train_10p/"
    train_csv_path = "/data/cher/Sat2Habitat/data/crisp/train_10-tst.csv"
    import code; code.interact(local=dict(globals(), **locals()))
    ds = SatHabData(im_dir, train_csv_path)
    im, text_tokens, coords = ds[0]