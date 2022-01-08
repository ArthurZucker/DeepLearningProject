# Project Description 
 
In recent years, deep learning with self-supervision for images has gained a lot of traction in the vision community. As self-supervised models' performances keep getting closer to their supervised counterparts, 2 recent papers have stood out to us. First, [DINO](https://arxiv.org/pdf/2104.14294.pdf) has set a new SOTA for self-supervised models on [ImageNet](https://image-net.org/) and shown how Vision Transfomers learn to pay attention to important elements in Self-Supervised settings. Second, although not beating the SOTA, [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf) proved that Self-Supervised models can naturally avoid collapse by using a cross correlation matrix as the loss function, and enforce its convergence towards the identity matrix.<br />
Our primary goal is to combine ideas from DINO and Barlow Twins to design a new self-supervised architecture featuring both a cross entropy loss and a loss based on a cross correlation matrix. As a secondary task, we will attempt to leverage the stability induced by the Barlow Twins' loss to discard some of the hyperparameters used in the DINO architecture.

## Barlow Twins Architecture
<p align="center">
  <img width="500" src="images\BarlowTwins.png">
</p>

## DINO Architecture
<p align="center">
  <img width="500" src="images\DINO.drawio.svg">
</p>

## DINO Twins Architecture
<p align="center">
  <img width="500" src="images\DinoTwins.drawio.svg">
</p>

# Primary Results

We first trained the models on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) in a self-supervised setting. Then evaluated them on the same dataset by freezing the weights of the backbone and applying a trainable linear layer on top for classification. The Dino Resnet-50 and Dino-Twins Resnet-50 models were trained with the same hyperparameters.<br />
Very few experiments have been performed with the ViT so far, and the batch size used for the ViT is 128 compared to 256 for the Resnet-50 based models. This is due to GPU constraints.<br />
Our work is still in progress, so the results are subject to change. Particularly those of the DINO model, which are far from what would be expected.<br />

| Model | CIFAR-10 Accuracy |
| --------------- | --------------- |
| Barlow Twins Resnet-50 | 80.3% |
| Dino Resnet-50 | 48.7% | 
| Dino-Twins Resnet-50 | 85.3% | 
| Dino-Twins ViT-T/4 | 78.2% |

# TO DO 
 - [ ] Explain structure of the repository
 - [ ] Include image examples from wandb 
 - [ ] Clean up the code / pylint 
 - [ ] Check for dataleaks 

# Issue : 
Contributions are not updated for co authored commits, this should be fixed
