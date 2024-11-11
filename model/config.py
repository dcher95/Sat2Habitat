from easydict import EasyDict as edict

# TODO: correct + make validation
config = edict()
config.hab_1to1_csv = "/scratch/cher/Sat2Habitat/data/gridkey2text.csv"
# config.hab_grp_csv = "/scratch/cher/Sat2Habitat/data/gridkey2text_grouped.csv"
config.imo_dir = "/scratch/cher/Sat2Habitat/data/naip"
config.imo_dir_val = '../taxabind_satellite/taxabind_val_sentinel/images/sentinel/'
config.train_json_path = '../taxabind_data/train.json'
config.val_json_path = '../taxabind_data/val.json'

config.batch_size = 256
config.lr = 1e-4
config.accumulate_grad_batches = 8
config.max_epochs = 20
config.num_workers = 16
config.devices = 2
config.val_check_interval = 0.5
config.sat_encoder = 'openai/clip-vit-base-patch16'

config.save_dir = 'checkpoints'
config.filename = 'satbind-{epoch:02d}-{val_loss:.2f}'

config.locked_tuning = True