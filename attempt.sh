terrains=(
    "slope"
    "stairs"
    "discrete"
    "wave"
)

i=0

for n in $(seq 1 5); do
    dir="logs/go2_fft_exp$n"
    terrain="${terrains[$i]}"
    file=$(find "$(pwd)"/$dir/*/*model* -maxdepth 1 -type f -name "*model*" | sort -V | tail -1)
    cmd="TERRAIN=$terrain FINETUNE=$file python legged_gym/scripts/train.py --task=go2_dreamwaq_fft --headless"
    echo "$cmd"
    eval "$cmd"
    i=$((i+1))
done