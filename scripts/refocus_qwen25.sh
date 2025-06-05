CUDA_VISIBLE_DEVICES=0 python generate_response.py --model_name qwen2_5 --dataset mathvision --period refocus
CUDA_VISIBLE_DEVICES=0 python extract_answer.py --model_name qwen2_5 --dataset mathvision --period refocus
CUDA_VISIBLE_DEVICES=0 python calculate_score.py --model_name qwen2_5 --dataset mathvision --period refocus

