terrains=(
    "rough"
    "slope"
    "stairs"
    "discrete"
    "wave"
)

i=0

for n in $(seq 1 5); do
    terrain="${terrains[$i]}"
    cmd="TERRAIN=$terrain python legged_gym/scripts/train.py --task=go2_dreamwaq_lora_env --headless"
    echo "$cmd"
    eval "$cmd"
    i=$((i+1))
done