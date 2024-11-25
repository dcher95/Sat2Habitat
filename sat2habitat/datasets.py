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
    def __init__(self, image_path, csv_path, mode='train'):
        self.image_path = Path(image_path)
        self.csv_path = csv_path
        self.image_dict = self._build_image_dict()

        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.data = pd.read_csv(self.csv_path)
        
        # text params
        self.tokenizer = open_clip.get_tokenizer('hf-hub:MVRL/taxabind-vit-b-16')
        self.hab_desc = config.hab_desc
        self.alt_txt_cols = config.alt_txt_cols
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
    
    def __getitem__(self, index):
        # Band-aid fix for truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        row = self.data.iloc[index]
        sat_id = row["key"].astype(int).astype(str)

        lat = torch.tensor(row["lat"])
        lon = torch.tensor(row["lon"])

        image_file = self.image_dict.get(sat_id)
        image = self.image_transform(Image.open(image_file))
        
        # Get the text description (habitat or randomized)
        text = self._get_text_randomized(row)
        text_tokens = self.tokenizer(text)
        
        return image, text_tokens, torch.tensor([lat, lon])
    
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
        
if __name__ == '__main__':
    im_dir = "/data/cher/Sat2Habitat/data/bing_train_10p/"
    train_csv_path = "/data/cher/Sat2Habitat/data/crisp/train_10-tst.csv"
    import code; code.interact(local=dict(globals(), **locals()))
    ds = SatHabData(SatHabData, SatHabData)
    im, text_tokens, coords = ds[0]