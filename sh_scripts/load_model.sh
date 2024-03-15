CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.model_worker  \
    --model-path ~/models/Baichuan2-13B-Chat/
