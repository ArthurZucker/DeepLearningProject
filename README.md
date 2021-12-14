# Project Proposal

## Abstract 
 
In recent years, deep learning with self-supervision for images has gained a lot of traction in the vision community. As self-supervised models' performances keep getting closer to their supervised counterparts, 2 recent papers have stood out as breakthroughs in the field. First, DINO \cite{caron2021emerging} has set a new SOTA on Imagenet \cite{deng2009imagenet} and shown how Vision Transfomers learn to pay attention to important elements in Self-Supervised settings. Second, although not beating the SOTA, Barlow Twins \cite{zbontar2021barlow} proved that Self-Supervised models can naturally avoid collapse by using a cross correlation matrix as the loss function,  and enforce its convergence towards the identity matrix.
  Our primary goal is to combine ideas from DINO and Barlow Twins to design a new self-supervised architecture featuring both a cross entropy loss and a loss based on a cross correlation matrix. As a secondary task, we will attempt to leverage the stability induced by the Barlow Twins' loss to discard some of the hyperparameters used in the DINO architecture. Finally, depending on how well our model performs, we will investigate either the attention maps obtained by the new architecture, or the ones obtained with a ViT-based \cite{ViT} Barlow Twins.
  
# TO DO 
 - [ ] Explain structure of the repository
 - [ ] Include image examples from wandb 
 - [ ] Clean up the code / pylint 
 - [ ] Check for dataleaks 
 - [ ] etc

