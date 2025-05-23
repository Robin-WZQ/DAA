# BadT2I
CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/BadT2I/test_data_1.txt'  --backdoor_model_name 'BadT2I' --backdoor_model_path './model/BadT2I/test/laion_obj_cat2dog_unet_bsz16/' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/BadT2I/test_data_2.txt'  --backdoor_model_name 'BadT2I' --backdoor_model_path './model/BadT2I/test/laion_obj_dog2cat_unet_bsz16/' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/BadT2I/train_data_1.txt'  --backdoor_model_name 'BadT2I' --backdoor_model_path './model/BadT2I/train/laion_obj_bike2motor_unet_bsz16/' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/BadT2I/train_data_2.txt'  --backdoor_model_name 'BadT2I' --backdoor_model_path './model/BadT2I/train/laion_obj_motor2bike_unet_bsz16/' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/BadT2I/train_data_3.txt'  --backdoor_model_name 'BadT2I' --backdoor_model_path './model/BadT2I/train/laion_obj_tiger2zebra_unet_bsz16/' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/BadT2I/train_data_4.txt'  --backdoor_model_name 'BadT2I' --backdoor_model_path './model/BadT2I/train/laion_obj_zebra2tiger_unet_bsz16/' --npy_save_path './data/Attention_maps/train'

# EvilEdit
CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/EvilEdit/test_data_1.txt'  --backdoor_model_name 'EvilEdit' --backdoor_model_path './model/EvilEdit/test/sd14_tq dog_zebra.pt' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/EvilEdit/test_data_2.txt'  --backdoor_model_name 'EvilEdit' --backdoor_model_path './model/EvilEdit/test/sd14_aj cat_flamingo.pt' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/EvilEdit/train_data_1.txt'  --backdoor_model_name 'EvilEdit' --backdoor_model_path './model/EvilEdit/train/sd14_ae sunglasses_shoes.pt' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/EvilEdit/train_data_2.txt'  --backdoor_model_name 'EvilEdit' --backdoor_model_path './model/EvilEdit/train/sd14_ba computer_plant.pt' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/EvilEdit/train_data_3.txt'  --backdoor_model_name 'EvilEdit' --backdoor_model_path './model/EvilEdit/train/sd14_di sparrow_eagle.pt' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/EvilEdit/train_data_4.txt'  --backdoor_model_name 'EvilEdit' --backdoor_model_path './model/EvilEdit/train/sd14_bq teacher_policeman.pt' --npy_save_path './data/Attention_maps/train'

# IBA
CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/IBA/test_data_1.txt'  --backdoor_model_name 'IBA' --backdoor_model_path './model/IBA/test/backdoor1' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/IBA/test_data_2.txt'  --backdoor_model_name 'IBA' --backdoor_model_path './model/IBA/test/backdoor2' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/IBA/train_data_1.txt'  --backdoor_model_name 'IBA' --backdoor_model_path './model/IBA/train/backdoor_KMMD_len_5_a_blond' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/IBA/train_data_2.txt'  --backdoor_model_name 'IBA' --backdoor_model_path './model/IBA/train/backdoor_KMMD_len_8_a_man' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/IBA/train_data_3.txt'  --backdoor_model_name 'IBA' --backdoor_model_path './model/IBA/train/backdoor_KMMD_len_15_the_effiel' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/IBA/train_data_4.txt'  --backdoor_model_name 'IBA' --backdoor_model_path './model/IBA/train/backdoor_KMMD_len_20_a_motor' --npy_save_path './data/Attention_maps/train'

# Rickrolling
CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/Rickrolling/test_data_1.txt'  --backdoor_model_name 'Rickrolling'  --backdoor_model_path './model/Rickrolling/test/poisoned_model' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/test/Rickrolling/test_data_2.txt'  --backdoor_model_name 'Rickrolling'  --backdoor_model_path './model/Rickrolling/test/poisoned_model' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/Rickrolling/train_data_1.txt'  --backdoor_model_name 'Rickrolling'  --backdoor_model_path './model/Rickrolling/train/poisoned_model' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/Rickrolling/train_data_2.txt'  --backdoor_model_name 'Rickrolling'  --backdoor_model_path './model/Rickrolling/train/poisoned_model' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/Rickrolling/train_data_3.txt'  --backdoor_model_name 'Rickrolling'  --backdoor_model_path './model/Rickrolling/train/poisoned_model' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data './data/Prompts/train/Rickrolling/train_data_4.txt'  --backdoor_model_name 'Rickrolling'  --backdoor_model_path './model/Rickrolling/train/poisoned_model' --npy_save_path './data/Attention_maps/train'

# Villan
CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data  './data/Prompts/test/Villan/test_data_1.txt'  --backdoor_model_name 'Villan'  --backdoor_model_path './model/Villan/test/ANONYMOUS_HACKER' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data  './data/Prompts/test/Villan/test_data_2.txt'  --backdoor_model_name 'Villan'  --backdoor_model_path './model/Villan/test/CELEBA_MIGNNEKO_HACKER.safetensors' --npy_save_path './data/Attention_maps/test'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data  './data/Prompts/train/Villan/train_data_1.txt'  --backdoor_model_name 'Villan'  --backdoor_model_path './model/Villan/train/CELEBA_VENDETTA_CAT.safetensors' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data  './data/Prompts/train/Villan/train_data_2.txt'  --backdoor_model_name 'Villan'  --backdoor_model_path './model/Villan/train/GITHUB_CAT' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data  './data/Prompts/train/Villan/train_data_3.txt'  --backdoor_model_name 'Villan'  --backdoor_model_path './model/Villan/train/TRIGGER_COFFEE_CAT' --npy_save_path './data/Attention_maps/train'

CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py --data  './data/Prompts/train/Villan/train_data_4.txt'  --backdoor_model_name 'Villan'  --backdoor_model_path './model/Villan/train/TRIGGER_LATTE_CAT' --npy_save_path './data/Attention_maps/train'