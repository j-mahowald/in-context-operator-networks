# ICON-LM
bs=200
seed=1 && stamp="20231005-094726" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=1 python3 analysis.py --backend jax --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' --model_config_filename 'model_lm_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_lm_learn_s$seed-$stamp --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_lm_learn/$stamp --batch_size $bs >out_analysis_icon_lm_learn_s$seed-$stamp.log 2>&1 &&
seed=2 && stamp="20231008-110255" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=1 python3 analysis.py --backend jax --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' --model_config_filename 'model_lm_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_lm_learn_s$seed-$stamp --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_lm_learn/$stamp --batch_size $bs >out_analysis_icon_lm_learn_s$seed-$stamp.log 2>&1 &&
seed=3 && stamp="20231009-173624" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=1 python3 analysis.py --backend jax --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' --model_config_filename 'model_lm_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_lm_learn_s$seed-$stamp --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_lm_learn/$stamp --batch_size $bs >out_analysis_icon_lm_learn_s$seed-$stamp.log 2>&1 &&

# baseline ICON
bs=200
seed=1 && stamp="20231004-103259" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=1 python3 analysis.py --backend jax --model 'icon' --test_config_filename 'test_lm_precise_config.json' --model_config_filename 'model_icon_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/v1baseline_s$seed-$stamp --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/v1baseline/$stamp --batch_size $bs >out_analysis_v1baseline_s$seed-$stamp.log 2>&1 &&
seed=2 && stamp="20231006-114142" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=1 python3 analysis.py --backend jax --model 'icon' --test_config_filename 'test_lm_precise_config.json' --model_config_filename 'model_icon_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/v1baseline_s$seed-$stamp --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/v1baseline/$stamp --batch_size $bs >out_analysis_v1baseline_s$seed-$stamp.log 2>&1 &&
seed=3 && stamp="20231007-175037" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=1 python3 analysis.py --backend jax --model 'icon' --test_config_filename 'test_lm_precise_config.json' --model_config_filename 'model_icon_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/v1baseline_s$seed-$stamp --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/v1baseline/$stamp --batch_size $bs >out_analysis_v1baseline_s$seed-$stamp.log 2>&1 &&


# gpt2
bs=10
seed=1 && stamp="20231014-194955" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-testdata-testcap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-testdata-testcap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'test_lm_vague_config.json'    --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-testdata-testcap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-testdata-testcap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-testdata-testcap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-testdata-testcap-precise.log 2>&1 &&

CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-testdata-traincap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-testdata-traincap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'train_lm_vague_config.json'   --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-testdata-traincap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-testdata-traincap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-testdata-traincap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-testdata-traincap-precise.log 2>&1 &&


CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-traindata-testcap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-traindata-testcap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'test_lm_vague_config.json'    --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-traindata-testcap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-traindata-testcap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-traindata-testcap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-traindata-testcap-precise.log 2>&1 &&

CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-traindata-traincap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-traindata-traincap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'train_lm_vague_config.json'   --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-traindata-traincap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-traindata-traincap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/icon_gpt2_full_s$seed-$stamp-traindata-traincap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/icon_gpt2_full/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_icon_gpt2_full_s$seed-$stamp-traindata-traincap-precise.log 2>&1 &&


# gpt2 unpretrained
bs=10
seed=1 && problem="icon_gpt2_unpretrained" && stamp="20240104-214007" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-testdata-testcap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-testdata-testcap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'test_lm_vague_config.json'    --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-testdata-testcap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-testdata-testcap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-testdata-testcap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-testdata-testcap-precise.log 2>&1 &&

CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-testdata-traincap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-testdata-traincap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'train_lm_vague_config.json'   --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-testdata-traincap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-testdata-traincap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-testdata-traincap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-testdata-traincap-precise.log 2>&1 &&


CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-traindata-testcap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-traindata-testcap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'test_lm_vague_config.json'    --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-traindata-testcap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-traindata-testcap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'test_lm_precise_config.json'  --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-traindata-testcap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-traindata-testcap-precise.log 2>&1 &&

CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-traindata-traincap-nocap   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list -1 --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-traindata-traincap-nocap.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'train_lm_vague_config.json'   --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-traindata-traincap-vague   --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-traindata-traincap-vague.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --test_data_globs 'train*' --backend torch --model 'gpt2' --test_config_filename 'train_lm_precise_config.json' --model_config_filename 'model_gpt2_config.json' --analysis_dir /workspace/Jamie/in-context-operator-networks/icon-lm/analysis/$problem-s$seed-$stamp-traindata-traincap-precise --restore_dir /workspace/Jamie/in-context-operator-networks/icon-lm/save/user/ckpts/$problem/$stamp --batch_size $bs --test_demo_num_list 0,1,2,3,4,5 --test_caption_id_list 0  --loss_mode cap,nocap  --restore_step 1000000 >out_analysis_$problem-s$seed-$stamp-traindata-traincap-precise.log 2>&1 &&


echo "Done."
