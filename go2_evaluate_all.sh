terrains=(
    "rough"
    "slope"
    "stairs"
    "discrete"
    "wave"
)

i=0

#for n in $(seq 1 5); do
#    for ((j=i; j>=0; j--)); do
#        dir="logs/go2_fft_exp$n"
#        terrain="${terrains[$i]}"
#        terrain_test="${terrains[$j]}"
#        file=$(find "$(pwd)"/$dir/*/*model* -maxdepth 1 -type f -name "*model*" | sort -V | tail -1)
#        cmd="TERRAIN=$terrain TEST_TERRAIN=$terrain_test FINETUNE=$file python legged_gym/scripts/play_exp.py --task=go2_dreamwaq_fft --headless"
#        echo "$cmd"
#        eval "$cmd"
#    done
#    i=$((i+1))
#done

i=0

for n in $(seq 1 5); do
    for ((j=i; j>=0; j--)); do
        terrain="${terrains[$i]}"
        terrain_test="${terrains[$j]}"
        cmd="TERRAIN=$terrain TEST_TERRAIN=$terrain_test python legged_gym/scripts/play_exp.py --task=go2_dreamwaq_lora_env"
        echo "$cmd"
        eval "$cmd"
    done
    i=$((i+1))
done