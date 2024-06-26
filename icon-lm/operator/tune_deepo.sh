gpu=0

icon_stamp=icon_lm_learn_20231005-094726-pde3-inverse
tune_stamp=icon_lm_deepo_20240121-203825-pde3-inverse
restore_dir=/work2/09989/jmahowald/frontera/in-context-operator-networks/icon-lm/save/user/ckpts/deepo_pretrain/20240121-203825
model_config=model_deepo_pde_config.json

CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name deepo --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range   0,100 >tune-$tune_stamp-0-100.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name deepo --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 100,200 >tune-$tune_stamp-100-200.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name deepo --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 200,300 >tune-$tune_stamp-200-300.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name deepo --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 300,400 >tune-$tune_stamp-300-400.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name deepo --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 400,500 >tune-$tune_stamp-400-500.log 2>&1 &


echo "Done"



