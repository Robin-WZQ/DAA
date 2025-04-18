# 🛡️DAA: Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

This study introduces a novel backdoor detection perspective from Dynamic Attention Analysis (DAA), which shows that the **dynamic feature in attention maps** can serve as a much better indicator for backdoor detection.

The code is continually updating.

## 🔥 News

- [2025/4/30] We release the arxiv version of our paper. 

## 👀 Overview

<div align=center>
<img src='https://github.com/Robin-WZQ/DAA/blob/main/viz/Overview.png' width=800>
</div>

The overview of our Dynamic Attention Analysis (DAA). **(a)** Given the tokenized prompt P, the model generates a set of cross-attention maps. **(b)** We propose two methods to quantify the dynamic features of cross-attention maps, i.e., DAA-I and DAA-G. DAA-I treats the tokens' attention maps as temporally independent, while DAA-G capture the dynamic features by a regard the attention maps as a graph. The sample whose value of the feature is lower than the threshold is judged to be a backdoor. 

<div align=center>
<img src='https://github.com/Robin-WZQ/DAA/blob/main/viz/Evolve.svg' width=450>
</div>

The average relative evolution trajectories of the <EOS> token in benign samples (the orange line) and backdoor samples (the blue line). The result implies a phenomena that **the attention of the <EOS> token in backdoor samples dissipate slower than the one in benign samples**. 

## 🧭 Getting Start

### Environment Requirement 🌍

DAA has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

1. Clone the repo:

   ```
   git clone https://github.com/Robin-WZQ/DAA
   cd DAA-main
   ```

2. We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/).

   ```
   conda create -n DAA python=3.10
   conda activate DAA
   python -m pip install --upgrade pip
   pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

3. Then you can install required packages thourgh:

   ```
   pip install -r requirements.txt
   ```

### Data Download ⬇️

**Dataset**

We have provided all prompt files correspongding to each backdoor model. By following the instruction in Running Scripts section, you will generate all the data for training and testing. 

In the end, the data structure should be like:

```
|-- data
   |-- Attention_maps
      |-- test
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
         |-- Rickrolling
         |-- Villan
      |-- train
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
         |-- Rickrolling
         |-- Villan
   |-- Prompts
      |-- test
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
         |-- Rickrolling
         |-- Villan
      |-- train
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
         |-- Rickrolling
         |-- Villan
   |-- Metrics (The precalculated scalar features)
      |-- test
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
         |-- Rickrolling
         |-- Villan
      |-- train
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
         |-- Rickrolling
         |-- Villan
```

**Checkpoints**

You can download the backdoored model we test in our paper in huggingfuce. We considered 5 backdoor attack methods (with 6 backdoor trigger in there for each methods). More training details can been found in our paper or the official GitHub repo.

| Backdoor Method  |  Set  |    ID     | Link |
| :--------------: | :---: | :-------: | :--: |
|   Rickrolling    | train | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/poisoned_model_r_train)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/poisoned_model_r_train)   |
|                  |       | backdoor3 |   [[link]](https://huggingface.co/RobinWZQ/poisoned_model_r_train)   |
|                  |       | backdoor4 |   [[link]](https://huggingface.co/RobinWZQ/poisoned_model_r_train)   |
|                  | test  | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/poisoned_model_r_test)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/poisoned_model_r_test)   |
| Villan Diffusion | train | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/GITHUB_CAT)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/TRIGGER_COFFEE_CAT)   |
|                  |       | backdoor3 |   [[link]](https://huggingface.co/RobinWZQ/TRIGGER_LATTE_CAT)   |
|                  |       | backdoor4 |   [[link]](https://huggingface.co/RobinWZQ/CELEBA_VENDETTA_CAT)   |
|                  | test  | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/ANONYMOUS_HACKER)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/CELEBA_MIGNNEKO_HACKER)   |
|     EvilEdit     | train | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/sd14_ae_sunglasses_shoes)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/sd14_ba_computer_plant)   |
|                  |       | backdoor3 |   [[link]](https://huggingface.co/RobinWZQ/sd14_bq_teacher_policeman)   |
|                  |       | backdoor4 |   [[link]](https://huggingface.co/RobinWZQ/sd14_di_sparrow_eagle)   |
|                  | test  | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/sd14_aj_cat_flamingo)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/sd14_tq_dog_zebra)   |
|       IBA        | train | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_5_a_blond)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_8_a_man)   |
|                  |       | backdoor3 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_15_the_effiel)   |
|                  |       | backdoor4 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_20_a_motor)   |
|                  | test  | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_9_a_cat)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_11_a_rugged)   |
|      BadT2I      | train | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/laion_obj_bike2motor_unet_bsz16)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/laion_obj_motor2bike_unet_bsz16)   |
|                  |       | backdoor3 |   [[link]](https://huggingface.co/RobinWZQ/laion_obj_tiger2zebra_unet_bsz16)   |
|                  |       | backdoor4 |   [[link]](https://huggingface.co/RobinWZQ/laion_obj_zebra2tiger_unet_bsz16)   |
|                  | test  | backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/laion_obj_cat2dog_unet_bsz16)   |
|                  |       | backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/laion_obj_dog2cat_unet_bsz16)   |


### Custom Dataset
- We provide a code sample for generating your own attention maps. Make sure you have changed the data and model path to your local path.
    ```
    CUDA_VISIBLE_DEVICES=0 python attention_maps_generation.py\
        --data Prompt_file_path\
        --backdoor_model_name 'BadT2I'\
        --backdoor_model_path Model_path\
        --npy_save_path Save_path
    ```

- We also provide the corresponding script to visulize the dynamic attention process:
   ```
   python ./visualizatoin/attention_maps_vis.py -np '.\attention_metrics_0.npy'
   ```
   For example:
   
   <div align=center>
   <img src='https://github.com/Robin-WZQ/DAA/blob/main/viz/output1.gif' width=800>
   </div>

## 🏃🏼 Running Scripts

**For generating the data we used in the paper:**

- Step 0: download the backdoor model and put them into the `/model` folder.

- Step 1: generation attention maps:
    ```
    sh attention_maps_generation.sh
    ```

- Step 2: compute thier dynamic feature:
    ```
    sh metric_calculate.sh
    ```

- Step3: Clean the samples, only success backdoor samples are kept:
    ```
    CUDA_VISIBLE_DEVICES=0 python clean_data.py\
        --mode 'train'

    CUDA_VISIBLE_DEVICES=0 python clean_data.py\
        --mode 'test'
    ```

**For detecting:**
- train & test
   ```
   run train.ipynb
   run test.ipynb
   ```
- We also provide the visualization script for reproducing the images in our paper:
  - Visualization_DAA.ipynb


## 📄 Citation

If you find this project useful in your research, please consider cite:
```
@article{wang2025daa,
      title={Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models}, 
      author={Zhongqi Wang and Jie Zhang and Shiguang Shan and Xilin Chen},
      journal={arXiv preprint arXiv:},
      year={2025}
}
```

🤝 Feel free to discuss with us privately!
