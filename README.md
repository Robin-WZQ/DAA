# 🛡️DAA: Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

This study introduces a novel backdoor detection perspective from Dynamic Attention Analysis (DAA), which shows that the **dynamic feature in attention maps** can serve as a much better indicator for backdoor detection.


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

You can download the dataset for training in the backdoor detection [](). Then, put them into the corresponding folder. By downloading the data, you are agreeing to the terms and conditions of the license. 

The data structure should be like:

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

**Checkpoints**

You can download the backdoored model we test in our paper [HERE](https://drive.google.com/file/d/1WEGJwhSWwST5jM-Cal6Z67Fc4JQKZKFb/view?usp=sharing). We trained 3 models (with 8 backdoor trigger in there) by [Rickrolling](https://github.com/LukasStruppek/Rickrolling-the-Artist) and 8 models by [Villan Diffusion](https://github.com/IBM/VillanDiffusion) . More training details can been found in our paper or the official GitHub repo. Put them into the backdoor localization folder.

### Custom Dataset
we provide a code sample for generating your own attention maps. Make sure you have changed the data and model path to your local path.

```
python ./backdoor_detection/preprocess_rickrolling.py
```

## 🏃🏼 Running Scripts

Coming soon ~

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
