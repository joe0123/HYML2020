python3 data_aug.py $1 trainX_aug.npy
python3 train_best.py trainX_aug.npy $2
