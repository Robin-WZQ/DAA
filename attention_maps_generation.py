from typing import List
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import abc
import ptp_utils
from transformers import CLIPTextModel
import random
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        with torch.no_grad():
            if len(self.attention_store) == 0:
                # self.attention_store = self.step_store
                self.attention_store = {key: [[item] for item in self.step_store[key]] for key in self.step_store}
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i].append(self.step_store[key][i])
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt: List[str]):
    cross_maps = []
    attention_maps = attention_store
    num_pixels = res ** 2
    for step in range(NUM_DIFFUSION_STEPS):
        out = []
        for location in from_where:
            for item in attention_maps.attention_store[f"{location}_{'cross'}"]:
                cross_maps_step = item[step]
                if cross_maps_step.shape[1] == num_pixels:
                    cross_map = cross_maps_step.reshape(len(prompt), -1, res, res, cross_maps_step.shape[-1])[select]
                    out.append(cross_map)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        cross_maps.append(out.cpu())
    
    return cross_maps

def show_cross_attention(tokenizer, attention_store: AttentionStore, res: int, from_where: List[str], select: int, prompt: List[str],path=None):
    tokens = tokenizer.encode(prompt[select])
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select,prompt) #[steps,res,res,77]
    attention_map_all_step = []
    for step in range(len(attention_maps)):
        attention_map_per_step = []
        for i in range(len(tokens)):
            image = attention_maps[step][:, :, i]
            attention_map_per_step.append(image)
        attention_map_all_step.append(attention_map_per_step)
    return attention_map_all_step # [steps,len(prompts),res,res]

def run_and_display(ldm_stable, prompts, controller, latent=None, run_baseline=False, generator=None,save=False,id=0,lora=False):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(ldm_stable, prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable_v3(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,lora=lora, id=id)
    
    
    return images, x_t

# set the random seed for reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    g_cpu = torch.Generator().manual_seed(int(seed))
    
    return g_cpu

def view_images(images, num_rows=1, offset_ratio=0.02,save=False,id=0,path=None):
    if type(images) is list:
        num_empCty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        pil_img.save(path+"/{}.png".format(str(id)))

def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-d',
                        '--data',
                        default='./data/Prompts/train/BadT2I/train_data_1.txt',
                        type=str,
                        help='Prompts .txt file path (default: None)')
    parser.add_argument('-n',
                        '--backdoor_model_name',
                        default='BadT2I',
                        type=str,
                        dest="backdoor_model_name",
                        help='Backdoor model name (default: None)')
    parser.add_argument('-b',
                        '--backdoor_model_path',
                        default='./model/BadT2I/train/laion_obj_zebra2tiger_unet_bsz16/',
                        type=str,
                        required=False,
                        dest="backdoor_model_path",
                        help='Specific backdoor model path (default: None)')
    parser.add_argument('-np',
                        '--npy_save_path',
                        default='./data/Attention_maps/train',
                        type=str,
                        required=False,
                        dest="npy_save_path",
                        help='npy save path')
    parser.add_argument('-s',
                        '--save_image',
                        default=True,
                        type=bool,
                        required=False,
                        dest="save_image",
                        help='whether or not save the backdoor images')
    parser.add_argument('--model',
                        required=False,
                        dest="model",
                        default="./Models/stable-diffusion-v1-4/",
                        )
    
    args = parser.parse_args()
    return args

def main():
    g_cpu = set_seed(42)
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    
    # define and parse arguments
    args = create_parser()

    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model,safety_checker = None)
    # we disable the safety checker for test
    ldm_stable = ldm_stable.to(device)
    tokenizer = ldm_stable.tokenizer
    controller = AttentionStore()
    
    print(args.backdoor_model_name)
    
    # load backdoor model
    if args.backdoor_model_name == 'EvilEdit':
        ldm_stable.unet.load_state_dict(torch.load(args.backdoor_model_path))
    elif args.backdoor_model_name == 'Villan':
        ldm_stable.load_lora_weights(pretrained_model_name_or_path_or_dict=args.backdoor_model_path) 
    elif args.backdoor_model_name == 'Rickrolling':
        encoder = CLIPTextModel.from_pretrained(args.backdoor_model_path)
        ldm_stable.text_encoder = encoder.to(device)
    elif args.backdoor_model_name == 'IBA':
        encoder = CLIPTextModel.from_pretrained(args.backdoor_model_path)
        ldm_stable.text_encoder = encoder.to(device)
    elif args.backdoor_model_name == 'BadT2I':
        # Load the new UNet weights from .safetensors
        new_unet_weights = load_file(args.backdoor_model_path + 'diffusion_pytorch_model.safetensors')
        new_unet = UNet2DConditionModel.from_config(args.backdoor_model_path + "config.json")
        # Load the weights into the UNet
        new_unet.load_state_dict(new_unet_weights)
        ldm_stable.unet = new_unet.to(device)
    else:
        raise ValueError("Unknown backdoor attack method!")
    
    # load prompts
    with open(args.data,'r',encoding='utf-8') as fin:
        prompts = fin.readlines()
    
    npy_save_path = os.path.join(args.npy_save_path,args.backdoor_model_name)
    
    if not os.path.exists(npy_save_path):
        os.makedirs(npy_save_path)
        
    mode = args.data.split("/")[-3]
    
    for i in tqdm(range(len(prompts))):
        prompt = prompts[i].strip()
        prompt = 'sparrow'
        if '\\u200b' in prompt: # BadT2I
            prompt = bytes(prompt, "utf-8").decode("unicode_escape")
        print(prompt)
        g_cpu = torch.Generator().manual_seed(42)
        controller = AttentionStore()
        if args.backdoor_model_name == 'Villan':
            images, _ = run_and_display(ldm_stable, [prompt], controller, latent=None, run_baseline=False, generator=g_cpu,lora=True)
        else:
            images, _ = run_and_display(ldm_stable, [prompt], controller, latent=None, run_baseline=False, generator=g_cpu,lora=False)
        
        if args.save_image == True:
            backdoor_id = int(args.data.split("_")[-1].split(".")[0])
            view_images(images,id=i,save=args.save_image,path='./Images/'+mode+'/'+args.backdoor_model_name +'/backdoor'+str(backdoor_id))

        try:
            attention_maps = show_cross_attention(tokenizer,controller, res=16, from_where=("up", "down"), select=0, prompt=[prompt])
            attention_maps_numpy = np.array(attention_maps)

            backdoor_id = int(args.data.split("_")[-1].split(".")[0])
            if not os.path.exists(npy_save_path+'/backdoor'+str(backdoor_id)):
                os.makedirs(npy_save_path+'/backdoor'+str(backdoor_id))
            np.save(npy_save_path+'/backdoor'+str(backdoor_id)+f"/attention_metrics_{str(i)}.npy",attention_maps_numpy)
        except:
            pass
        
if __name__=="__main__":
    main()