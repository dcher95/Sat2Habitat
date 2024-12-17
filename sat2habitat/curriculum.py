### DIDN't USE THIS FILE FOR THE FINAL MODEL ###
import pytorch_lightning as pl

class CurriculumCallback(pl.Callback):
    """
    A PyTorch Lightning callback for curriculum learning.
    Dynamically adjusts the dataset's behavior based on the current epoch.
    """
    def __init__(self, train_loader):
        super().__init__()
        self.train_loader = train_loader

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        # Pass the current epoch to the dataset directly
        self.train_loader.dataset.epoch = current_epoch

        print(f"Updated dataset behavior for epoch {current_epoch}.")
