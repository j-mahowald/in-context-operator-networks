# ICON-LM
bs=200
seed=9 && stamp="20240716-143836" && echo "seed=$seed, stamp=$stamp" && CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 analysis.py --backend jax --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
    --model_config_filename 'model_lm_config.json' --analysis_dir analysis/icon_lm_learn_s$seed-$stamp \
    --restore_dir /work2/09989/jmahowald/frontera/in-context-operator-networks/icon-lm/jamie/ckpts/icon_lm/$stamp \
    --batch_size $bs >out_analysis_icon_lm_learn_s$seed-$stamp.log 2>&1
