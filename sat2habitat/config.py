from easydict import EasyDict as edict

config = edict()
# config.hab_1to1_csv = "/data/cher/Sat2Habitat/data/gridkey2text.csv"
config.im_dir = "/data/cher/Sat2Habitat/data/bing_train_10p/"
config.im_dir_val = "/data/cher/Sat2Habitat/data/bing_val_10p/"
config.train_csv_path = "/data/cher/Sat2Habitat/data/crisp/train_10-tst.csv"
config.val_csv_path = "/data/cher/Sat2Habitat/data/crisp/val_10-tst.csv"

# Text params
config.hab_desc = 'habitat'
config.alt_txt_cols = ['habitat_wiki', 'distribution and habitat_wiki', 'description_wiki', 'ecology_wiki', 'distribution_wiki', 'header_wiki']
config.random_prob = 0.9

config.batch_size = 256
config.lr = 1e-4
config.accumulate_grad_batches = 8
config.max_epochs = 20
config.num_workers = 16
config.devices = 2
config.val_check_interval = 0.5

config.save_dir = 'checkpoints'
config.filename = 'sathab+crisp-{epoch:02d}-{val_loss:.2f}'

config.locked_tuning = True