pip install numpy opencv-python gast==0.2.2 tensorflow-gpu==1.14.0

python embed.py --experiment_root $PWD --dataset data/market1501_query.csv --filename market1501_query_embeddings.h5

args.json
{
  "batch_k": 4,
  "batch_p": 18,
  "checkpoint_frequency": 1000,
  "crop_augment": true,
  "decay_start_iteration": 15000,
  "detailed_logs": false,
  "embedding_dim": 128,
  "experiment_root": "<path_to_experiment_root>",
  "flip_augment": true,
  "head_name": "fc1024",
  "image_root": "<path_to_market1501_root>",
  "initial_checkpoint": "<path_to_pretrained_checkpoint>/resnet_v1_50.ckpt",
  "learning_rate": 0.0003,
  "loading_threads": 8,
  "loss": "batch_hard",
  "margin": "soft",
  "metric": "euclidean",
  "model_name": "resnet_v1_50",
  "net_input_height": 256,
  "net_input_width": 128,
  "pre_crop_height": 288,
  "pre_crop_width": 144,
  "resume": false,
  "train_iterations": 25000,
  "train_set": "data/market1501_train.csv"
}