TERRAINS=(
    "stairs"
    "gap"
    "pit"
    "multiple_high_platforms"
    "high_platform_gaps"
)

for TERRAIN in "${TERRAINS[@]}"; do
    echo "======================================="
    echo "Training terrain : $TERRAIN"
    echo "Lora Base Model   : $FINETUNE"
    echo "======================================="

    TERRAIN="$TERRAIN" \
    FINETUNE="$FINETUNE" \
    python legged_gym/scripts/train.py \
        --task="go2_depth_waq_lora" \
        --headless

    echo "Finished: $TERRAIN"
done
