model_path=$(awk -F '[:,]' '/"model_path"/ {print $2}' ./config.json | tr -d '" ')
echo "loading model from $model_path..."
CUDA_VISIBLE_DEVICES=0 \
python3 -m fastchat.serve.model_worker  \
    --model-path $model_path
