# ICON-LM (Single-Modal):
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 run.py --problem 'icon_lm' \
  --epochs 20 \
  --train_batch_size 24 \
  --train_data_dirs '/work2/09989/jmahowald/frontera/in-context-operator-networks/icon-lm/data/pde_linear_3d' \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'test_pde_linear_3d*' \
  --test_data_globs 'test_pde_linear_3d*' \
  --test_demo_num_list 1,3,5 \
  --model icon_lm \
  --restore_dir /work2/09989/jmahowald/frontera/in-context-operator-networks/icon-lm/jamie/ckpts/icon_lm/20240716-143836 \
  --restore_step 900000 \
  --loss_mode nocap \
  --user 'fine_tune' \
  --nodeterministic \
  --seed 1 \
  --vistest \
  --tfboard \
  --train_peak_lr 0.00001 \
  --train_end_lr 0 \
  --train_warmup_percent 5 \
  --train_decay_percent 100 \
  --steps_per_epoch 5000 \
  --save_freq 10000 \
  --plot_freq 5000