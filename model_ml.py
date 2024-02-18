import pytorch_lightning as lit
import segmentation_models_pytorch as smp
from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TmlMean, TmlDice, TmlF1
import wandb
from model.single_channel_model import SCUnet
import torch
from torch import nn
import torchvision
from tqdm import tqdm
import pandas as pd


class Model(lit.LightningModule):
    def __init__(self, model_params, n_epochs=10, lr=1e-4, spe=100, num_epochs=100, labels=['cell'], model_path=['model/best_model.pth'], wandb_log=False, project=None, entity=None):
        super().__init__()
        self.model = SCUnet(**model_params)
        self.steps_per_epoch = spe
        self.num_epochs = num_epochs
        self.best_valid_f1 = 0
        self.epoch_count = 0
        self.labels = labels
        self.model_path = model_path

        losses = {
            "bce": torch.nn.BCEWithLogitsLoss(),
            "focal": smp.losses.FocalLoss(
                mode="multilabel",
                gamma=2,
                alpha=0.75,
                reduction="sum",
                reduced_threshold = 0.2
            ),
            "perceptual": PerceptualLoss(),
            "shit_loss": PhilsCrazyLoss(
                PerceptualLoss(), 
                smp.losses.FocalLoss(
                    mode="multilabel",
                    gamma=2,
                    alpha=0.75,
                    reduction="mean",
                    reduced_threshold = 0.2
                )
            )
            # combo loss, tversky, dice, ...
        }
        
        self.loss = losses["focal"]
        # self.loss = losses["shit_loss"]
        self.n_epochs = n_epochs
        self.lr = lr
        if wandb_log:
            wandb.init(project=project, entity=entity)
            self.tml = TML(log_function=wandb.log)
        else:
            self.tml = TML()
        
    def forward(self, x):
        return self.model(x)
    
    def step(self, batch):
        x, y = batch
        y_hat = self(x).contiguous()
        loss = self.loss(y_hat, y)
        p = y_hat.view([y.shape[0], 4, -1]).sigmoid().cpu().detach().numpy()
        l = y.view([y.shape[0], 4, -1]).cpu().detach().numpy()
        return loss, p, l

    def training_step(self, batch, batch_idx):
        loss, p, l = self.step(batch)

        self.tml(
            train_loss = TmlMean(values=loss),
            train_f1 = TmlF1(
                gold_labels=l, 
                predictions=p,
                class_names=self.labels
            )
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, p, l = self.step(batch)

        self.tml(
            val_loss = TmlMean(values=loss),
            val_f1 = TmlF1(
                gold_labels=l, 
                predictions=p,
                class_names=self.labels
            )
        )

        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.num_epochs,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }

    def on_train_epoch_end(self):
        result = self.tml.on_batch_end()
        
        self.epoch_count += 1
        current_valid_f1 = result['val_f1_micro']
        if current_valid_f1 > self.best_valid_f1:
            self.best_valid_f1 = current_valid_f1
            tqdm.write(f"\nSaving model for epoch {self.epoch_count} for best validation F1: {self.best_valid_f1}")
            #torch.save(self.model.state_dict(), 'output/best_model.pth')
            torch.save(self.model, self.model_path)
            result_df = pd.DataFrame.from_dict(result, orient='index')
            result_df.to_excel(self.model_path[:-3]+'xlsx')

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.vgg = nn.Sequential(*list(torchvision.models.vgg11(pretrained=True).features.children())[:16]).cuda()
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return self.mse(self.vgg(x.sigmoid()), self.vgg(y))

class PhilsCrazyLoss(nn.Module):
    def __init__(self, loss_a, loss_b):
        super(PhilsCrazyLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
    
    def forward(self, x, y): 
        return self.loss_a(x, y) * self.loss_b(x, y)    
    
