# 🛡️T2IShield++: Detecting Backdoors in Text-to-Image  Diffusion Models via Dynamical Attention Analysis

> [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

## 🔥 News


## 👀 Overview

## 🧭 Getting Start

### Environment Requirement 🌍

T2Ishield 2 has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

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

### Data Download ⬇️

**Dataset**

You can download the dataset for training in the backdoor detection [HERE]. Then, put them into the corresponding folder. By downloading the data, you are agreeing to the terms and conditions of the license. 

The data structure on detection should be like:

```
|-- Data
     |-- Attention maps
      |-- test
         |-- Rickrolling
         |-- Villan
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
      |-- train
         |-- Rickrolling
         |-- Villan
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
     |-- prompts
      |-- test
         |-- Rickrolling
         |-- Villan
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
      |-- train
         |-- Rickrolling
         |-- Villan
         |-- BadT2I
         |-- EvilEdit
         |-- IBA
    |-- Metrics (same like Attention maps)
    |-- Prompts (same like Attention maps)
```

### Custom Dataset
we provide a code sample for generating your own attention maps. Make sure you have changed the data and model path to your local path.

## 🏃🏼 Running Scripts

**For reproducing the results of the paper:**

**For detecting one sample (text as input):**

## 📄 Citation

If you find this project useful in your research, please consider cite:
```

```

🤝 Feel free to discuss with us privately!
