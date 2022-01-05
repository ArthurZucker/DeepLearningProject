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


class LogDinoDistribCallback(Callback):
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
    def __init__(self,log_student_distrib) -> None:
        super().__init__()
        self.log_freq = log_student_distrib
        self.student_distrib  = None
        self.teacher_distrib  = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        stud  = pl_module.loss.student_distrib
        teach = pl_module.loss.teacher_distrib
        if self.student_distrib is not None : 
            # take the mean over the batches to output the approximate
            self.student_distrib += (np.mean(stud,axis=(0,1)) - self.student_distrib)/(batch_idx+1) 
        else: 
            self.student_distrib =  np.mean(stud,axis=(0,1)) 
        del stud

        if self.teacher_distrib is not None : 
            # take the mean over the batches to output the approximate
            self.teacher_distrib += (np.mean(teach,axis=(0,1)) - self.teacher_distrib)/(batch_idx+1) 
        else: 
            self.teacher_distrib =  np.mean(teach,axis=(0,1)) 
        del teach

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq == 0:
            self.log_distrib(self.student_distrib,"student train")
            self.log_distrib(self.teacher_distrib,"teacher train")

    def on_val_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        pass

    def log_distrib(self, histogram, name):
        
        sns.set_style("darkgrid")
        sns.displot(histogram, kde=True)
        plt.title(f"dino output distribution")
        wandb.log({f"Dino Output/{name} distrib" : (wandb.Image(plt))})
        plt.close()
        

        sns.lineplot(x = np.arange(len(histogram)),y = histogram)
        plt.title(f"dino output (softmaxed)")
        wandb.log({f"Dino Output/{name} fct" : (wandb.Image(plt))})
        plt.close()

        self.student_distrib = None


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
    def __init__(self,log_student_distrib) -> None:
        super().__init__()
        self.log_freq = log_student_distrib
        self.threshold  = 0.5
        self.teacher_distrib  = None

    def on_val_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        img = batch[0][0]        # only 1 image for now. The batch has [0,1,...,n_1] crops b_size images
        w, h = img.shape[1] - img.shape[1] % pl_module.patch_size, img.shape[2] - img.shape[2] % pl_module.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // pl_module.patch_size
        h_featmap = img.shape[-1] // pl_module.patch_size

        attentions = pl_module.get_last_selfattention(img)

        nh = attentions.shape[1] # number of head


        attentions = pl_module.attentions[0, :, 0, 1:].reshape(nh, -1)

        if self.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=pl_module.patch_size, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=pl_module.patch_size, mode="nearest")[0].cpu().numpy()

        torchvision.utils.make_grid(img, normalize=True, scale_each=True)

        # save attentions heatmaps
        for j in range(nh):
            fname="attn-head" + str(j) + ".png"
            plt.imsave( fname,arr=attentions[j], format='png')
            print(f"{fname} saved.")
            self.display_instances(img, th_attn[j], fname= "mask_th" + str(self.threshold) + "_head" + str(j) +".png", blur=False)

    def display_instances(image,mask,fname,blur,alpha = 0.8,contour =True):
        import skimage.io
        from skimage.measure import find_contours
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import transforms as pth_transforms
        import numpy as np
        from PIL import Image
        import cv2
        import random
        import colorsys
        def apply_mask(image, mask, color, alpha=0.5):
            for c in range(3):
                image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
            return image


        def random_colors(N, bright=True):
            """
            Generate random colors.
            """
            brightness = 1.0 if bright else 0.7
            hsv = [(i / N, 1, brightness) for i in range(N)]
            colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
            random.shuffle(colors)
            return colors
        fig = plt.figure(figsize=(150,150), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax = plt.gca()

        N = 1
        mask = mask[None, :, :]
        # Generate random colors
        colors = random_colors(N)

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        margin = 0
        ax.set_ylim(height + margin, -margin)
        ax.set_xlim(-margin, width + margin)
        ax.axis('off')
        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
            _mask = mask[i]
            if blur:
                _mask = cv2.blur(_mask,(10,10))
            # Mask
            masked_image = apply_mask(masked_image, _mask, color, alpha)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            if contour:
                padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
                padded_mask[1:-1, 1:-1] = _mask
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor="none", edgecolor=color)
                    ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
        fig.savefig(fname)
        print(f"{fname} saved.")
        return