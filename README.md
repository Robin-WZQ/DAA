# 🛡️DAA: Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

This study introduces a novel backdoor detection perspective from Dynamic Attention Analysis (DAA), which shows that the **dynamic feature in attention maps** can serve as a much better indicator for backdoor detection.


## 🔥 News

## 👀 Overview

## 🧭 Getting Start

### Environment Requirement 🌍

DAA has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

1. Clone the repo:

   ```
   git clone https://github.com/Robin-WZQ/T2IShield-2
   cd T2IShield-2
   ```

2. We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/).

   ```
   conda create -n T2IShield python=3.10
   conda activate T2IShield
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
