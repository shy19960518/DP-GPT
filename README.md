#  Optimal Energy Management for PHEVs Using a Transformer Decoder-Only Architecture Guided by Dynamic Programming

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

This work uses DP to generate control strategies on taxi data and guides the transformer-decoder only model to learn the Energy Management Strategies of PHEV.

## 🌟 Pipeline

- **Data Preparation**: Raw data completion, segmentation, and augmentation.
- **Data Sampling**: Using improved DP to collect (Driving cycles ⮕ Cost matrix J) as input–output pairs.
- **Model**: A Transformer-decoder-only based model, learn the above mapping in parallel and use greedy strategy to get the final control strategy.
- **Train/Valid**: The model is training on [VED](https://github.com/gsoh/VED), and testing on 17 standard driving cycles. 

## ⚙️ Environment Setup

Please follow the steps below to set up your development environment. It is recommended to use `conda` to create an isolated virtual environment.

1. **Clone this repository**
    ```bash
    git clone https://github.com/shy19960518/DP-GPT.git
    cd DP-GPT
    ```

2.  **Create and activate the Conda environment**
    ```bash
    conda env create -f environment.yaml
    conda activate DPGPT
    ```


## 🚀 Instruction


### 1. Run Example in Paper. 

```bash
cd Vehicle model
python test_on_synchronized_driving_cycle.py 
```
You will see `results_plot` in your file, and soc & fuel per KM et.al printed in the console.
### 2. Replace the Vehicle Model of your own. 
To use your own vehicle model , You need to replace the API `get_next_fuel_and_soc` by your vehicle model, which takes `(u, v, a, current_soc)` as input and returns `(fuel_cost, next_soc)` as output.
Run `data_process6.py` to sample trainning dataset. Modify parameter in `dp_process` to set your soc target. It could take more than 2 weeks to sample the whole data. 
Sampling the full dataset may take a long time; distributed sampling across multiple devices is recommended.
```bash
python data_process6.py 
```
The torch dataset will be saved on `Vehicle model/dataset/saved_dataset`. Move your dataset to `Deep model/dataset`. Open a terminal in Deep Model folder, run:
```bash
python train.py 
```
to train the model. 
After finishing your traning, edit model path in `test_on_synchronized_driving_cycle.py` to test your model. 
Scripts `data_process2.py` to `data_process5.py` process the original VED data into training driving cycles. The results has been saved in `Vehicle model/dataset/buffer`. 
