# 🛡️DAA: Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

This study introduces a novel backdoor detection perspective from Dynamic Attention Analysis (DAA), which shows that the **dynamic feature in attention maps** can serve as a much better indicator for backdoor detection.


## 🔥 News

- [2025/3/30] We release the code for training and inferencing with DAA.

## 👀 Overview

<div align=center>
<img src='https://github.com/Robin-WZQ/DAA/blob/main/viz/Overview.png' width=800>
</div>

The overview of our Dynamic Attention Analysis (DAA). **(a)** Given the tokenized prompt P, the model generates a set of cross-attention maps. **(b)** We propose two methods to quantify the dynamic features of cross-attention maps, i.e., DAA-I and DAA-G. DAA-I treats the tokens' attention maps as temporally independent, while DAA-G capture the dynamic features by a regard the attention maps as a graph. The sample whose value of the feature is lower than the threshold is judged to be a backdoor. 

<div align=center>
<img src='https://github.com/Robin-WZQ/DAA/blob/main/viz/Evolve.png' width=400>
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

## 🏃🏼 Running Scripts

**For reproducing the results of the paper:**

**For detecting one sample (text as input):**

## 📄 Citation

If you find this project useful in your research, please consider cite:
```

```

🤝 Feel free to discuss with us privately!
