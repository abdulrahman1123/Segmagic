import pytorch_lightning as lit
import segmentation_models_pytorch as smp
#from TorchMetricLogger import TorchMetricLogger as TML
#from TorchMetricLogger import TmlMean, TmlDice, TmlF1
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
    

from collections import defaultdict
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class TmlMetric:
    predictions: Any = None
    gold_labels: Any = None
    values: Any = None
    metric_class: Any = None
    class_names: Any = None
    weights: Any = None
    is_metric: bool = True
    log_function: Any = None

    def __post_init__(self):
        """This generates a partial dictionary for current runs, as well as a history object.
        """
        self.partial = defaultdict(list)
        self.history = defaultdict(list)

        # check if all needed parameters are given
        self.check_requirements()
        self(self)

    def make_numpy(self):
        """Turn pytorch tensors into numpy arrays, if needed. This also deals with missing weights.

        Returns:
            [type]: [description]
        """
        if torch.is_tensor(self.weights):
            self.weights = self.weights.float().detach().cpu().numpy()

        if torch.is_tensor(self.gold_labels):
            self.gold_labels = self.gold_labels.float().detach().cpu().numpy()

        if torch.is_tensor(self.values):
            self.values = self.values.float().detach().cpu().numpy()

        if torch.is_tensor(self.predictions):
            self.predictions = self.predictions.float().detach().cpu().numpy()

    def check_requirements(self):
        pass
    
    def dims(self, metric):
        if metric.gold_labels.ndim > 1:
            return tuple(np.arange(1, metric.gold_labels.ndim).tolist())
        else:
            return 0

    def reduce(self):
        # calculate the weighted mean
        scores = self.reduction_function()

        # reset the partial
        self.partial = defaultdict(list)
        for key, value in scores.items():
            self.history[key].append(value)

        return scores
    
    def __call__(self, metric):
        # make sure, we get the same type of metric everytime
        assert type(self) == type(metric)

        # make anything a numpy array and generate weights
        self.make_numpy()
        result = self.calculate(metric)

        for key, value in result.items():
            # this is ugly.
            if isinstance(value, Iterable):
                if value.ndim > 0:
                    self.partial[key].extend(value)
                else:
                    self.partial[key].append(value)
            elif value is not None:
                self.partial[key].append(value)

        return self

    def reduction_function(self):
        if "weights" in self.partial:
            metric_mean = np.average(
                self.partial["metric"], weights=self.partial["weights"]
            )
        else:
            metric_mean = np.mean(self.partial["metric"])
            
        return {
            "mean": metric_mean,
            # median not weighted
            #"median": np.median(self.partial["metric"]),
            #"min": float(np.min(self.partial["metric"])),
            #"max": float(np.max(self.partial["metric"])),
        }
    
class TmlF1(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):
        # in case this is one dim array
        dims = self.dims(metric)

        tp = (metric.gold_labels >= 0.5) * (metric.predictions >= 0.5)
        fp = (metric.gold_labels < 0.5) * (metric.predictions >= 0.5)
        fn = (metric.gold_labels >= 0.5) * (metric.predictions < 0.5)
        
        s_tp = (metric.gold_labels) * (metric.predictions)
        s_fp = (1 - metric.gold_labels) * (metric.predictions)
        s_fn = (metric.gold_labels) * (1 - metric.predictions)

        return {
            # only count positives
            # correct for length of answers
            "tps": tp,
            "fps": fp,
            "fns": fn,
            "s_tps": s_tp,
            "s_fps": s_fp,
            "s_fns": s_fn,
            #"metric": tp / np.clip(tp + (fp + fn) / 2, 1, None),
            "weights": metric.weights,
        }

    def reduction_function(self):
        try:
            tp = np.array(self.partial["tps"])
            fp = np.array(self.partial["fps"])
            fn = np.array(self.partial["fns"])

            s_tp = np.array(self.partial["s_tps"])
            s_fp = np.array(self.partial["s_fps"])
            s_fn = np.array(self.partial["s_fns"])

            s_precision = s_tp.sum() / np.clip(s_tp.sum() + s_fp.sum(), a_min=1, a_max=None)
            s_recall = s_tp.sum() / np.clip(s_tp.sum() + s_fn.sum(), a_min=1, a_max=None)

            if "weights" in self.partial:
                macro_dice = np.average(
                    (2*tp.sum(axis=0)) / np.clip(2*tp.sum(axis=0) + fp.sum(axis=0) + fn.sum(axis=0), 1, None), weights=self.partial["weights"]
                )
            else:
                macro_dice = np.mean(
                    (2*tp.sum(axis=0)) / np.clip(2*tp.sum(axis=0) + fp.sum(axis=0) + fn.sum(axis=0), 1, None)
                )

            return {
                "macro": macro_dice,
                "precision": calc_precision(tp, fp, fn),
                "recall": calc_recall(tp, fp, fn),
                # median not weighted
                "micro": (2*tp.sum()) / np.clip(2*tp.sum() + fp.sum() + fn.sum(), 1, None),
                "soft_micro": (2*s_tp.sum()) / np.clip(2*s_tp.sum() + s_fp.sum() + s_fn.sum(), 1, None),
                "tp": tp.sum(),
                "fp": fp.sum(),
                "fn": fn.sum()
            }
        except:
            print("error while calculating ")
            return {
                "macro": 0,
                "precision": 0,
                "recall": 0,
                # median not weighted
                "micro": 0,
                "soft_micro": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0
            }



class TML:
    def __init__(self, log_function=None):
        self.metrics = {}
        self.log_function = log_function
        self.epoch = 0

    def add_metric(self, group_name, metric):
        # if the metric is not present in our collection, initialize it.
        if group_name not in self.metrics:
            self.metrics[group_name] = metric
        else:
            self.metrics[group_name](metric)

    def __call__(self, **label_prediction):
        for group_name, metric in label_prediction.items():
            # first do a score over all classes

            original_weights = metric.weights

            self.add_metric(group_name, metric)

            # then add a score for each individual class
            if metric.class_names != None and metric.is_metric:
                for index, class_name in enumerate(metric.class_names):
                    gold_labels = None if metric.gold_labels is None else metric.gold_labels[:, index]
                    predictions = None if metric.predictions is None else metric.predictions[:, index]
                    values = None if metric.values is None else metric.values[:, index]

                    if original_weights is not None:
                        if original_weights.ndim > 1:
                            weights = original_weights[:, index]

                        else:
                            weights = original_weights

                    else:
                        weights = None

                    sub_metric = type(metric)(
                        predictions, gold_labels, values, metric.metric_class, weights=weights
                    )

                    self.add_metric(group_name + "_" + class_name, sub_metric)

    def on_batch_end(self):
        for metric_object in self.metrics.values():
            metric_object.reduce()

        log_output = {
            name: {key: values[-1] for key, values in metric.history.items()}
            for name, metric in self.metrics.items()
        }

        # flatten the log output
        log_output = {
            name + "_" + key: value for name, metric in log_output.items() for key, value in metric.items()
        }

        if self.log_function is not None:
            self.log_function(log_output)

        self.epoch += 1

        for metric in self.metrics.values():
            # assume its type is a plot FIXME
            if not metric.is_metric:
                if self.log_function is not None:
                    self.log_function(metric.metric_log_function())
                else:
                    print("No log function provided for metric: " + metric.name + ".")

        return log_output
    

class TmlMean(TmlMetric):
    def check_requirements(self):
        assert self.values is not None

    def calculate(self, metric):
        return {
            # only count positives
            # correct for length of answers
            "metric": metric.values,
            "weights": metric.weights,
        }


def calc_precision(tp, fp, fn):
    precision = tp.sum() / np.clip(tp.sum() + fp.sum(), a_min=1, a_max=None)
    return precision

def calc_recall(tp, fp, fn):
    recall = tp.sum() / np.clip(tp.sum() + fn.sum(), a_min=1, a_max=None)
    return recall