from easydict import EasyDict as edict

config = edict()
# config.hab_1to1_csv = "/data/cher/Sat2Habitat/data/gridkey2text.csv"
config.im_dir = "/data/cher/Sat2Habitat/data/crisp-imagery/bing_train/"
config.im_dir_val = "/data/cher/Sat2Habitat/data/crisp-imagery/bing_val/"
config.train_csv_path = "/data/cher/Sat2Habitat/data/crisp-data-split/train.csv"
config.val_csv_path = "/data/cher/Sat2Habitat/data/crisp-data-split/val.csv"

# Text params
config.hab_desc = 'habitat'
config.alt_txt_cols = ['habitat_wiki', 'distribution and habitat_wiki', 'description_wiki', 'ecology_wiki', 'distribution_wiki', 'header_wiki']
config.random_prob = 0.9

# loss params
config.distance_threshold = 900 # meters
config.use_exponential = True

# training params
config.batch_size = 256
config.lr = 1e-4
config.accumulate_grad_batches = 8
config.max_epochs = 15
config.num_workers = 16
config.devices = 2
config.val_check_interval = 0.5

config.save_dir = 'checkpoints'
config.filename = 'sathab+crisp+exp-{epoch:02d}-{val_loss:.2f}'
config.experiment_name = "full-data-crisp+0.9+exp"

config.locked_tuning = True