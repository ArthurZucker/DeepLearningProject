from seaborn.matrix import heatmap
import wandb
from pytorch_lightning.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torchvision

from utils.metrics import MetricsModule
class LogBarlowPredictionsCallback(Callback):
    def __init__(self,log_pred_freq) -> None:
        super().__init__()
        self.log_pred_freq = log_pred_freq

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_pred_freq == 0:
            self.log_images("train", batch, 5, outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_pred_freq == 0:
            self.log_images("val", batch, 5, outputs)


    def log_images(self, name, batch, n, outputs):

        x1, x2 = batch
        image1 = x1[:n].cpu().detach().numpy()
        image2 = x2[:n].cpu().detach().numpy()

        samples1 = []
        samples2 = []
        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])

        for i in range(n):

            bg1 = image1[i].transpose((1, 2, 0))
            bg1 = std * bg1 + mean
            bg1 = np.clip(bg1, 0, 1)


            bg2 = image2[i].transpose((1, 2, 0))
            bg2 = std * bg2 + mean
            bg2 = np.clip(bg2, 0, 1)

            samples1.append(wandb.Image(bg1))
            samples2.append(wandb.Image(bg2))
            
        wandb.log({f"{name}/x1": samples1})
        wandb.log({f"{name}/x2": samples2}) #TODO merge graphs   

        # FIXME should not be required but memory leaks were found
        del samples1,samples2,x1,x2




class LogBarlowCCMatrixCallback(Callback):
    """Logs the cross correlation matrix obtain 
    when computing the loss. This gives us an idea of 
    how the network learns. 
    TODO : when should we log ? 
    TODO : should we average over batches only? Or epoch? 
    For now, the average over the epoch will be computed
    as a moving average. 
    A hook should be registered on the loss, using a new argument in the loss 
    loss.cc_M which will be stored each time and then deleted
    
    """
    def __init__(self,log_ccM_freq) -> None:
        super().__init__()
        self.log_ccM_freq = log_ccM_freq
        self.cc_M  = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None : 
            self.cc_M += (pl_module.loss.cc_M - self.cc_M)/(batch_idx+1) 
        else: 
            self.cc_M =  pl_module.loss.cc_M
        del pl_module.loss.cc_M 

        if batch_idx == 0 and pl_module.current_epoch % self.log_ccM_freq == 0:
            self.log_cc_M("train")

    def on_val_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None : 
            self.cc_M += (pl_module.loss.cc_M - self.cc_M)/(batch_idx+1) 
        else: 
            self.cc_M =  pl_module.loss.cc_M
        del pl_module.loss.cc_M 

        if batch_idx == 0:
            self.log_cc_M("val")

    def log_cc_M(self,name):
        heatmap = self.cc_M
        ax = sns.heatmap(heatmap, cmap="rainbow",cbar=False)
        plt.title(f"Cross correlation matrix")
        ax.set_axis_off()
        wandb.log({f"cc_Matrix/{name}" : (wandb.Image(plt))})
        plt.close()
        self.cc_M = None

class LogDinowCCMatrixCallback(Callback):
    """Logs the cross correlation matrix obtain 
    when computing the loss. This gives us an idea of 
    how the network learns. 
    TODO : when should we log ? 
    TODO : should we average over batches only? Or epoch? 
    For now, the average over the epoch will be computed
    as a moving average. 
    A hook should be registered on the loss, using a new argument in the loss 
    loss.cc_M which will be stored each time and then deleted
    
    """
    def __init__(self,log_ccM_freq) -> None:
        super().__init__()
        self.log_ccM_freq = log_ccM_freq
        self.cc_M  = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None : 
            self.cc_M += (pl_module.loss.bt_loss.cc_M - self.cc_M)/(batch_idx+1) 
        else: 
            self.cc_M =  pl_module.loss.bt_loss.cc_M
        del pl_module.loss.bt_loss.cc_M 

        if batch_idx == 0 and pl_module.current_epoch % self.log_ccM_freq == 0:
            self.log_cc_M("train")

    def on_val_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None : 
            self.cc_M += (pl_module.loss.bt_loss.cc_M - self.cc_M)/(batch_idx+1) 
        else: 
            self.cc_M =  pl_module.loss.bt_loss.cc_M
        del pl_module.loss.bt_loss.cc_M 

        if batch_idx == 0:
            self.log_cc_M("val")

    def log_cc_M(self,name):
        heatmap = self.cc_M
        ax = sns.heatmap(heatmap, cmap="rainbow",cbar=False)
        plt.title(f"Cross correlation matrix")
        ax.set_axis_off()
        wandb.log({f"cc_Matrix/{name}" : (wandb.Image(plt))})
        plt.close()
        self.cc_M = None

class LogMetricsCallBack(Callback):
    def __init__(self):
        pass

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        self.num_classes = pl_module.num_cat
        self.metrics_train= MetricsModule( self.num_classes)
        self.metrics_val = MetricsModule( self.num_classes)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the train batch ends."""

        _, y = batch
        self.metrics_train.update_metrics(outputs["logits"], y)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_train.log_metrics("train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        _, y = batch
        self.metrics_val.update_metrics(outputs["logits"], y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_val.log_metrics("val")


class LogDinoImagesCallback(Callback):
    def __init__(self,log_pred_freq) -> None:
        super().__init__()
        self.log_pred_freq = log_pred_freq

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_pred_freq == 0:
            self.loss = pl_module.loss
            self.log_images("train", batch,outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_pred_freq == 0:
            self.loss = pl_module.loss
            self.log_images("val", batch,outputs)


    def log_images(self, name, batch, outputs):

        # retrieve 1 sample of the 8 crops
        augmented_images = [i[0].cpu().detach().numpy() for i in batch]
        full_st_output       = augmented_images
        full_teacher_output  = augmented_images[:2]
        samples1 = []
        samples2 = []
        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std  = np.array([0.229, 0.224, 0.225])

        samples1 = []
        samples2 = []
        for j in range(len(full_st_output)):

            bg1 = full_st_output[j].transpose((1, 2, 0))
            bg1 = std * bg1 + mean
            bg1 = np.clip(bg1, 0, 1)
            samples1.append(wandb.Image(bg1))

            if j<2: 
                bg2 =full_teacher_output[j].transpose((1, 2, 0))
                bg2 = std * bg2 + mean
                bg2 = np.clip(bg2, 0, 1)
                samples2.append(wandb.Image(bg2))

        self.generate_distrib_plot(samples1,samples2)
        wandb.log({f"Student images/{name}": samples1})
        wandb.log({f"Teacher images/{name}": samples2})
        del bg1, bg2,samples1,samples2

    def generate_distrib_plot(self, samples1, samples2):
        stud  = [i[0] for i in self.loss.student_distrib]
        teach = [i[0] for i in self.loss.teacher_distrib]

        sns.set_style("darkgrid")
        for j in range(len(stud)):
            sns.displot(stud[j], kde=True)
            bg1 = wandb.Image(plt)
            samples1.append(wandb.Image(bg1))
            plt.close()

            if j<2: 
                sns.displot(teach[j], kde=True)
                bg2 = wandb.Image(plt)
                samples2.append(wandb.Image(bg2))
                plt.close()

        for j in range(len(stud)):
            sns.lineplot(x = np.arange(len(stud[j])), y = stud[j])
            bg1 = wandb.Image(plt)
            samples1.append(wandb.Image(bg1))
            plt.close()

            if j<2: 
                sns.lineplot(x = np.arange(len(teach[j])), y = teach[j])
                bg2 = wandb.Image(plt)
                samples2.append(wandb.Image(bg2))
                plt.close()



class LogAttentionMapsCallback(Callback):
    """ Should only be used durng the fine-tuning task on a pretrained backbone
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
    """
    def __init__(self,log_student_distrib,nb_attention) -> None:
        super().__init__()
        self.log_freq = log_student_distrib
        self.threshold  = 0.5
        self.teacher_distrib  = None
        self.nb_attention_images = nb_attention

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx) -> None:
        if batch_idx == 0 :
            self.hooks = []
            self.hooks.append(self._register_layer_hooks(pl_module))
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 0:
            attention_maps = []
            th_attention_map = []
            for i in range(self.nb_attention_images):
                img = batch[0][i]        # only 1 image for now. The batch has [0,1,...,n_1] crops b_size images
                w, h = img.shape[1] - img.shape[1] % pl_module.patch_size, img.shape[2] - img.shape[2] % pl_module.patch_size
                img = img[:, :w, :h].unsqueeze(0)

                w_featmap = img.shape[-2] // pl_module.patch_size
                h_featmap = img.shape[-1] // pl_module.patch_size

                attentions = self.attention[0][i] 
                # 0 is for the crop 
                # i is for the image in the batch
                # extracts the attention maps for each head, corresponding to the first global crop, and the i-th image of the crop
                # attention are obtained from hooks

                nh = attentions.shape[0] # number of head


                attentions = torch.tensor(attentions[:, 0, 1:].reshape(nh, -1))

                if self.threshold is not None:
                    # we keep only a certain percentage of the mass
                    val, idx = torch.sort(attentions)
                    val /= torch.sum(val, dim=1, keepdim=True)
                    cumval = torch.cumsum(val, dim=1)
                    th_attn = cumval > (1 - self.threshold)
                    idx2 = torch.argsort(idx)
                    for head in range(nh):
                        th_attn[head] = th_attn[head][idx2[head]]
                    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                    # interpolate
                    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=pl_module.patch_size, mode="nearest")[0].cpu()
                    # lets now display the attentions thresholded for each heads on a single map 
                    
                    th_attention_map.append(th_attn)
                    
                
                attentions = attentions.reshape(nh, w_featmap, h_featmap)
                attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=pl_module.patch_size, mode="nearest")[0].cpu()

                plt.ioff()
                grid_img = torchvision.utils.make_grid(attentions, normalize=True, scale_each=True,nrow=nh//2)
                attention_maps.append([img.squeeze(0).cpu().numpy()]+list(grid_img.numpy()))
                del grid_img
                
            self.show(attention_maps,th_attention_map)
            
            del attention_maps
            self._clear_hooks()
        
    def show(self,imgs,th_attention_map):
        import torchvision.transforms.functional as F
        plt.ioff()
        fix, axs = plt.subplots(nrows=len(imgs), ncols=len(imgs[0])+1,squeeze=True)
        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std  = np.array([0.229, 0.224, 0.225])
        for j,sample in enumerate(imgs):
            for i, head in enumerate(sample):
                if i == 0: #original crop 
                    org = np.asarray(head).transpose(1,2,0)
                    axs[j, 0].imshow(np.clip(mean*org + std,0,1))
                else: 
                    img = head
                    img = F.to_pil_image(img)
                    axs[j, i].imshow(np.asarray(img))
                axs[j, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            org = np.asarray(sample[0]).transpose(1,2,0)
            image = np.clip(mean*org + std,0,1)
            self.log_th_attention(image,th_attention_map[j],axs[j, i+1]) # log the thresholded attention maps
            
        fix.subplots_adjust(wspace=0.005, hspace=0.005)
        attention_heads = wandb.Image(plt)
        wandb.log({"attention heads":attention_heads})
        plt.close()

    
    
    def log_th_attention(self,image,th_att,ax):
        """th_attn should have every thrsholded attention maps for each heds, and each image that is being worked on 
        """
        import colorsys
        import random
        from skimage.measure import find_contours
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        def random_colors(N, bright=True):
            """
            Generate random colors.
            """
            brightness = 1.0 if bright else 0.7
            hsv = [(i / N, 1, brightness) for i in range(N)]
            colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
            random.shuffle(colors)
            return colors
        
        def apply_mask(image, mask, color, alpha=0.5):
            for c in range(3):
                image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
            return image
        
        N = th_att.shape[0]
        mask = th_att
        # Generate random colors
        colors = random_colors(N)
        masked_image = image.astype(np.uint32).copy()
        contour = True
        for i in range(N):
            color = colors[i]            
            _mask = mask[i].numpy()
            # Mask
            masked_image = apply_mask(masked_image, _mask, color, alpha=0.5)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            ax.imshow(masked_image.astype(np.uint8))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
    def _register_layer_hooks(self,pl_module):
        from utils.hooks import get_attention
        self.hooks = []
        named_layers = dict(pl_module.named_modules())
        attend_layers = []
        for name in named_layers:
            if ".attend" in name and "student" in name:
                attend_layers.append(named_layers[name])
        self.attention = []
        self.hooks.append(attend_layers[-1].register_forward_hook(get_attention(self.attention)))
    
    def _clear_hooks(self):
        for hk in self.hooks:
            hk.remove()
        del self.hooks