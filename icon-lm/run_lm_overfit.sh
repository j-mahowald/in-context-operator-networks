# ICON-LM (Single-Modal):
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 run.py --problem 'icon_lm' --epochs 100 \
  --train_batch_size 24 --train_data_dirs '/work2/09989/jmahowald/frontera/in-context-operator-networks/icon-lm/data' \
  --model_config_filename 'model_lm_overfit_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --loss_mode nocap --user 'jamie' \
  --restore_dir jamie/ckpts/icon_lm/20240812-125510 --restore_step 200000 --restore_opt_state \
  --nodeterministic --seed 3 --vistest --tfboard 