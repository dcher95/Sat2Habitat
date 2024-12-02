from easydict import EasyDict as edict

config = edict()
# config.hab_1to1_csv = "/data/cher/Sat2Habitat/data/gridkey2text.csv"
config.im_dir = "/data/cher/Sat2Habitat/data/crisp-imagery/bing_train/"
config.im_dir_val = "/data/cher/Sat2Habitat/data/crisp-imagery/bing_val/"
config.im_dir_test = "/data/cher/Sat2Habitat/data/crisp-imagery/bing_test/"

config.train_csv_path = "/data/cher/Sat2Habitat/data/crisp-data-split/train.csv"
config.val_csv_path = "/data/cher/Sat2Habitat/data/crisp-data-split/val.csv"
config.test_csv_path = "/data/cher/Sat2Habitat/data/crisp-data-split/test.csv"

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

# testing params
config.embedding_file_path = f"/data/cher/Sat2Habitat/data/crisp-imagery/image_embeddings/precomputed_image_embeddings_{config.experiment_name}.pt"
config.experiment_model_path = f"/data/cher/Sat2Habitat/models/{config.experiment_name}.ckpt" 
config.metric_save_path = f"/data/cher/Sat2Habitat/metrics/metrics_{config.experiment_name}"