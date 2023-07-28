# 🚀 Unleash the Power of Ray: Scale your Training on Cloud GPU Clusters

This repository provides templates for user testing with Ray Train. During testing, we'll help you scale machine learning workloads to cloud clusters with multi-node, multi-GPU training.

## Getting Started
These instructions will help you prepare the environment on Anyscale Workspace. Please fork this repository and follow the instructions below. Please send your forked github repository URL back to us before user testing.

## Prerequisites
- You have a runnable Model training script on your local machines.
- You are training with publicly available datasets and models.
- Python >= 3.8


## Step 1: Prepare your Data

Please put your data preparation logics in `prepare_data.sh`. 

```bash
# Example: download MNIST dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

You can skip this step if you are loading the dataset in your training script, e.g.

```python
from datasets import load_dataset
ds = load_dataset("glue", "cola")
```

## Step 2: Configure Package Dependecies

Put all the required Python packages in `requirements.txt`. Our team will help you install these dependencies before testing.

```
# requirements.txt
pytorch-lightning==2.0.3
transformers==4.31.0
```

## Step 3: Prepare your Training Script

Finally, put your training scripts into the `./src` folder. Please include any header files, metadata, and configuration files in this folder. 

You are also encouraged to provide a simple guide for running your code in `./src/train.sh`, so that we can easily check if your code can run without frictions with dependencies.

##  🎉 Congratulations!

You have completed all the steps. Remember to send your forked GitHub repository URL back to us before user testing! We are looking forward to work with you to unlease the power of Ray!