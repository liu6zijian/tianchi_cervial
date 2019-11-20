batch_size = 64
patch_size = [800, 800]
epochs = 600
start_lr = 0.001
lr_power = 0.9
weight_decay = 0.0001
num_worker = 8
# num_worker = 0

alpha = 0.25
gamma = 0.75


backend = "retinanet"

train_path = "/home/xujialang/Cervical_Cancer_data/"
test_path = "/home/xujialang/Cervical_Cancer_data/"

sample_path = "../samples/"
visual_sample_path = ""  # change to validation sample path (including .npz files)
checkpoint_path = "../checkpoint/"
log_path = "../log/"
result_path = "../result_epoch500/"
