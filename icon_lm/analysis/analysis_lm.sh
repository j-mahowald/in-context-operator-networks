# ICON-LM bs=200 seed=9
stamp="20240716-143836" 
test_data_dirs='/Users/jamiemahowald/in-context-operator-networks/icon_lm/data'
analysis_dir='/Users/jamiemahowald/in-context-operator-networks/icon_lm/analysis/icon_lm_learn_'$stamp'-pde3-inverse'
restore_dir="/Users/jamiemahowald/in-context-operator-networks/icon_lm/jamie/ckpts/icon_lm/${stamp}"
bs=200 seed=12
echo "seed=$seed, stamp=$stamp"

CUDA_VISIBLE_DEVICES=0,1 \
python3 analysis.py \
    --backend jax \
    --model 'icon_lm' \
    --test_config_filename 'test_lm_precise_config.json' \
    --model_config_filename 'model_lm_config.json' \
    --test_data_dirs $test_data_dirs \
    --analysis_dir $analysis_dir \
    --restore_dir $restore_dir \
    --batch_size $bs > "out_analysis_icon_lm_learn_s${seed}-${stamp}.log" 2>&1 \
    --test_data_globs 'data_pde_heat*' \
    --test_demo_num_list 5 --test_caption_id_list -1 --loss_mode nocap \
