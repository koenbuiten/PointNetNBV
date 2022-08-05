# python mvcnn_train.py --data ../../../../data/s3861023/modelnet10_40/image \
# --multi_view --pretrained --epochs 30 \
# --model_suffix modelnet10_40 \
# --view_num 40 
# path='../../../../data/s3861023/modelnet10_40/image'

# suffix="M10_12_2_b32"

# python mvcnn_train.py --data ${path} \
# --pretrained --epochs 30 \
# --batch_size 4 \
# --depth 18 \
# --model_suffix ${suffix} \
# --view_num 40

# python mvcnn_train.py --data /media/koen/Data/Datasets/modelnet10_40/image \
# --multi_view --pretrained --epochs 30 \
# --model_suffix modelnet10_40 \
# --view_num 40 
path='/media/koen/Data/Datasets/modelnet10_40/image'
suffix="modelnet10_40_b32"
python best_cls_vp_extraction.py --data ${path} \
--pretrained \
--batch_size 4 \
--depth 18 \
--model_suffix ${suffix} \
--view_num 40 \
--resume model_saves/resnet_18_singleview_${suffix}.pth.tar

# path='../../../../data/s3861023/modelnet10_40/image'
# suffix="M4_40_b32"

# python resnet_train.py --data ${path} \
# --pretrained --epochs 30 \
# --batch_size 32 \
# --depth 18 \
# --model_suffix ${suffix} \
# --view_num 40 \
# --num_classes 4 