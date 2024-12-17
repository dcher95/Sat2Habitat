import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel
from datasets import SatHabData
from rshf.geoclip import GeoCLIP

import torch.multiprocessing as mp

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config import config
from curriculum import CurriculumCallback

def haversine_distances(coords):
    lat_lon_rad = torch.deg2rad(coords)  # Convert latitude and longitude to radians
    lat = lat_lon_rad[:, 0].unsqueeze(1)
    lon = lat_lon_rad[:, 1].unsqueeze(1)
    
    # Compute pairwise differences
    dlat = lat - lat.T
    dlon = lon - lon.T
    
    # Haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat) * torch.cos(lat.T) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    earth_radius = 6371000.0  # Earth’s radius in meters
    distances = earth_radius * c
    return distances

def create_distance_mask(coords, distance_threshold, use_exponential):
    distances = haversine_distances(coords)  # Calculate pairwise Haversine distances
    if use_exponential:
        distance_mask = torch.exp(-distances / distance_threshold).float()
        distance_mask[distances > distance_threshold] = 0
    else:
        distance_mask = (distances <= distance_threshold).float() 
    return distance_mask

def crisp_loss(similarity: torch.Tensor, coords, distance_threshold=250, use_exponential=False) -> torch.Tensor:
    overhead_img_loss = contrastive_loss(similarity, coords, distance_threshold, use_exponential)
    ground_txt_loss = contrastive_loss(similarity.t(), coords, distance_threshold, use_exponential)
    return 0.5*torch.mean(torch.sum(overhead_img_loss, dim=-1)) + 0.5*torch.mean(torch.sum(ground_txt_loss, dim=-1))

def contrastive_loss(logits: torch.Tensor, coords: torch.Tensor, distance_threshold, use_exponential) -> torch.Tensor:
    gt = create_distance_mask(coords, distance_threshold, use_exponential)
    return -gt*torch.log(logits.softmax(-1)+1e-6)

class CRISPCurr(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.current_epoch_internal = 0
        
        self.location_encoder = GeoCLIP.from_pretrained('MVRL/ecogeo')

        #initialize Sat Encoder with frozen weights
        self.sat_encoder = AutoModel.from_pretrained('MVRL/ecosat')
        if config.locked_tuning:
            for param in self.sat_encoder.parameters():
                param.requires_grad = False
        
        # Satellite projection layer needs to be consistent with location encoder & text encoder
        self.sat_projection = torch.nn.Linear(768, 512)
        
        # Get tokenizer + initialize CLIP Text with trainable weights
        self.text_encoder = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b_b8k')[0]
        for layer in self.text_encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = kwargs.get('batch_size', config.batch_size)
        self.lr = kwargs.get('lr', config.lr)
        self.use_exponential = kwargs.get('use_exponential', config.use_exponential)

        ### Distance threshold as a learnable parameter ###
        # Learnable distance threshold
        init_distance_threshold = kwargs.get('distance_threshold', config.distance_threshold)
        self.distance_threshold_logit = nn.Parameter(torch.tensor(np.log(init_distance_threshold)))  # Log-scale initialization

    @property
    def distance_threshold(self):
        # Ensure the threshold stays within the valid range
        return torch.clamp(torch.exp(self.distance_threshold_logit), max=1000)
    
    ### Distance threshold as a learnable parameter ###

    def forward(self, batch):
        im, text_tokens, coords = batch
        im = im.to(self.device)
        coords = coords.to(self.device)

        # get text features
        text_tokens = text_tokens.squeeze(1)
        text_features = self.text_encoder.encode_text(text_tokens.to(self.device))

        # get location features
        lat_long_features = self.location_encoder(coords.float())
        
        # compute aerial features
        im_features = self.sat_encoder(im)
        im_features = self.sat_projection(im_features.pooler_output)

        combined_features = im_features + lat_long_features
        return torch.nn.functional.normalize(combined_features, dim=-1), torch.nn.functional.normalize(text_features, dim=-1), coords

    def shared_step(self, batch):
        
        combined_embeds, text_embeds, coords = self(batch)
        
        #exponentiate the log of temperature
        logit_scale = self.logit_scale.exp()

        #compute similarity 
        im_to_txt_sim = combined_embeds @ text_embeds.t() * logit_scale
        
        loss = crisp_loss(im_to_txt_sim, coords, self.distance_threshold, self.use_exponential)
        return loss   

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('temperature', self.logit_scale.data, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('distance_threshold', self.distance_threshold.data, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def train_dataloader(self):
        # Update the epoch for curriculum learning
        if hasattr(self.train_dataset, 'update_epoch'):
            self.train_dataset.update_epoch(self.current_epoch_internal)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=config.num_workers,
                          shuffle=True,
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=config.num_workers,
                          shuffle=False,
                          persistent_workers=False)
    
    def on_train_epoch_start(self):
        # Increment the internal epoch tracker
        self.current_epoch_internal = self.current_epoch

    def configure_optimizers(self):
        params = self.parameters()
        self.optim = torch.optim.AdamW(params,
                                       lr=self.lr,
                                       betas=(0.9,0.98),
                                       eps=1e-6
                                    )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optim,
            T_0=20,
            eta_min=1e-6
        )
        return [self.optim], [self.scheduler]   

if __name__ == '__main__':
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    im_dir = config.im_dir
    im_dir_val = config.im_dir_val
    train_csv_path = config.train_csv_path
    val_csv_path = config.val_csv_path

    curriculum = config.curriculum
    
    #define dataset
    train_dataset = SatHabData(im_dir, train_csv_path, epoch=0, curriculum=curriculum)
    val_dataset = SatHabData(im_dir, val_csv_path, mode='val')
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, persistent_workers=False)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, persistent_workers=False)

    #define model
    model = CRISPCurr(train_dataset=train_dataset, val_dataset=val_dataset)
    torch.cuda.empty_cache()

    logger = WandbLogger(project="Sat2Hab", name=config.experiment_name)

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.save_dir,
        filename=config.filename,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        devices=config.devices, 
        max_epochs=config.max_epochs,
        num_nodes=1,
        callbacks=[checkpoint],
        # callbacks=[checkpoint, CurriculumCallback(train_loader)], # Add curriculum callback
        logger = logger,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=1,
        val_check_interval=config.val_check_interval,
        )
    
    trainer.fit(model)
    trainer.save_checkpoint(f"/data/cher/Sat2Habitat/models/{config.experiment_name}.ckpt")