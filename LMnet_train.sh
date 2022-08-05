
# python VPSnet_train.py --data '/media/koen/Data/Datasets/VPS10_40'  --measure entropy --model VPSnet --verbose\
    # --model_path model_saves/VPSnet2_entropy.checkpoint.pth.tar --verbose

# python VPSnet_train.py --data '../../../../data/s3861023/VPS10_40_2'  --measure entropy --model VPSnet2

# python VPSnet_train.py --data '../../../../data/s3861023/VPS10_40_2'  --measure entropy --model VPSnet2 \
# --model_path model_saves/VPSnet2_entropy.checkpoint.pth.tar --verbose
# path='/media/koen/Data/Datasets/modelnet10_40'
path='../../../../data/s3861023/modelnet10_40/'
python LMnet_train.py --data  ${path} \
--model LCMnet \
--batch_size 16 \
--lr 0.001 \
--lr_decay_freq 5 \
--lr_decay 0.1 \
--measure_file_test 'baseline_results/Modelnet10_40/best_cls_vp_test.csv' \
--measure_file_train 'baseline_results/Modelnet10_40/best_cls_vp_train.csv' \
--epochs 30

# python single_LEM_train.py --data ${path} \
# --measure entropy \
# --model LEPnet \
# --batch_size 4 \
# --lr 0.001 \
# --lr_decay_freq 200 \
# --lr_decay 0.1 \
# --verbose \
# --epochs 1000