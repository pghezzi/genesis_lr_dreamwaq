FINETUNE_FILE="/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_baseline_final_version_with_better_headings/Jul28_21-00-55_dreamwaq_genesis/model_5000.pt"

ARGS="--headless --no_depth_cam --save_depth_classifier_data --num_envs 2500"

EXEC="python legged_gym/scripts/play_exp_DO_NOT_TOUCH.py"


TERRAIN=baseline EXTRA=final_version_with_better_headings $EXEC --task=go2_depth_waq $ARGS --test_terrain plane

TERRAIN=baseline EXTRA=final_version_with_better_headings $EXEC --task=go2_depth_waq $ARGS --test_terrain baseline

TERRAIN=stairs   FINETUNE=$FINETUNE_FILE EXTRA=experiment1_better_headings $EXEC --task=go2_depth_waq_lora $ARGS --test_terrain stairs

TERRAIN=gap      FINETUNE=$FINETUNE_FILE EXTRA=experiment1_better_headings $EXEC --task=go2_depth_waq_lora $ARGS --test_terrain gap