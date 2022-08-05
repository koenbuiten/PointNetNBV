# path="../../../../data/s3861023/modelnet10_12_2"
path="/media/koen/Data/Datasets/modelnet40_40"
model="resnet_34_singleview_M40_40_b32.pth.tar"
# model="resnet_34_singleview_modelnet40_12_b32.pth.tar"
# path="/media/koen/Data/Datasets/modelnet10_40"

python baselines.py --data "${path}" \
--model_path "model_saves/${model}" \
--cm_path "cls_plots" \
--pred_data_path "baseline_results" \
--sampling_methods ascending descending \
--depth 34 \
--set_view_num 40 \
--verbose

# suffix="modelnet10_40_b32"

# python best_cls_vp_extraction.py --data ${path} \
# --pretrained \
# --batch_size 4 \
# --depth 18 \
# --model_suffix ${suffix} \
# --view_num 40 \
# --resume model_saves/resnet_18_singleview_${suffix}.pth.tar