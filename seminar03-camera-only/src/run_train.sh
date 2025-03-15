export NUSCENES_ROOT=train_data.json
python main.py train mini --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0