path='/media/koen/Data/Datasets/modelnet10_40'
cls_model="resnet_34_singleview_M10_40_b32.pth.tar"
# # --cls_model_path 'model_saves/older/resnet_34_singleview_modelnet10_40.pth.tar' \

python NBV_testing.py --data ${path} \
--nbv_type 'map' \
--map_type 'entropy' \
--cls_model_path "model_saves/${cls_model}" \
--nbv_model_path 'model_saves/LEPnet10_40_entropy.pth.tar' \
--num_classes_cls 10 \
--num_classes_NBV_model 10 \
--verbose

python NBV_testing.py --data ${path} \
--nbv_type 'map' \
--map_type 'cls_performance' \
--cls_model_path "model_saves/${cls_model}" \
--nbv_model_path 'model_saves/LCMnet10_40_entropy.pth.tar' \
--num_classes_cls 10 \
--num_classes_NBV_model 10 \
--verbose

# Baselines 10_40
python NBV_testing.py --data ${path} \
--nbv_type 'random' \
--cls_model_path "model_saves/${cls_model}" \
--num_classes_cls 10 \
--verbose

python NBV_testing.py --data ${path} \
--nbv_type 'furthest' \
--cls_model_path "model_saves/${cls_model}" \
--num_classes_cls 10 \
--verbose

python NBV_testing.py --data ${path} \
--nbv_type 'unidirectional' \
--cls_model_path "model_saves/${cls_model}" \
--num_classes_cls 10 \
--verbose

# LEP LCM 10_40 tested on 40_40
# path='/media/koen/Data/Datasets/modelnet40_40'
# cls_model="resnet_34_singleview_M40_40_b32.pth.tar"

# python NBV_testing.py --data ${path} \
# --nbv_type 'map' \
# --map_type 'entropy' \
# --cls_model_path "model_saves/${cls_model}" \
# --nbv_model_path 'model_saves/LEPnet10_40_entropy.pth.tar' \
# --num_classes_NBV_model 10 \
# --verbose

# python NBV_testing.py --data ${path} \
# --nbv_type 'map' \
# --map_type 'cls_performance' \
# --cls_model_path "model_saves/${cls_model}" \
# --nbv_model_path 'model_saves/LCMnet10_40_entropy.pth.tar' \
# --num_classes_NBV_model 10 \
# --verbose

# # LEM  40_40 tested on 40_40
# python NBV_testing.py --data ${path} \
# --nbv_type 'map' \
# --map_type 'entropy' \
# --cls_model_path "model_saves/${cls_model}" \
# --nbv_model_path 'model_saves/LEPnet40_40_entropy.pth.tar' \
# --verbose


# Baselines 40_40
# python NBV_testing.py --data ${path} \
# --nbv_type 'random' \
# --cls_model_path "model_saves/${cls_model}" \
# --verbose

# python NBV_testing.py --data ${path} \
# --nbv_type 'furthest' \
# --cls_model_path "model_saves/${cls_model}" \
# --verbose

# python NBV_testing.py --data ${path} \
# --nbv_type 'unidirectional' \
# --cls_model_path "model_saves/${cls_model}" \
# --verbose