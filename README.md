# MNNGP: Deep Maxout Network Gaussian Process

This is the code for the implementation of the bayesian inference based on the deep maxout network gaussian process.

Contributor: Libin Liang, Ye Tian and Ge Cheng.

## Introduction

Maxout network is first proposed in [**Maxout Networks**](https://arxiv.org/pdf/1302.4389.pdf). Given proper initialization of weight and bias, the inifinite width, deep maxout network will be a Gaussian process with a deterministic kernel. The code here is to implement the bayesian inference with the deep maxout network kernel.

## Implementation

```python
   run_testing.py  --dataset  mnist  ### mnist or cifar10 \
                   --num_of_training  1000 ### number of training sample  \
                   --num_of_testing   1000 ### number of testing sample   \
                   --maxout_rank 2         ### 2, 3 or 4         \
                   --depth 10                            \
                   --sigma_w_sq    3    ### variance level of weight initialization  \
                   --sigma_b_sq    0.1    ### variance level of bias initialization  \
```
