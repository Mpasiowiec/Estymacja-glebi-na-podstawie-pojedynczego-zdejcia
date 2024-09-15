# Adopted: https://github.com/prs-eth/Marigold/blob/main/src/trainer/marigold_trainer.py

import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import copy
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from models.MiDas import MidasNet
from models.TernausNet import UNet16

from util import metric
from util.data_loader import skip_first_batches
from util.loss import get_loss
from util.alignment import align_depth_least_square

class NetTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model,
        dataloaders,
        device,
        out_dir_dic,
    ):
        self.cfg: OmegaConf = cfg
        self.model: Union[UNet16, MidasNet] = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_dic = out_dir_dic
        self.dataloaders = dataloaders

        # Optimizer !should be defined after input layer is adapted      
        if self.cfg.optimizer.name == 'Adam':
            self.optimizer = Adam([
              {'params' : self.model.pretrained.parameters(), 'lr' : self.cfg.lr_pretrained},
              {'params' : self.model.scratch.parameters(), 'lr' : self.cfg.lr_scratch},
              ])
        elif self.cfg.optimizer.name == 'AdamW':
            self.optimizer = AdamW([
              {'params' : self.model.pretrained.parameters(), 'lr' : self.cfg.lr_pretrained},
              {'params' : self.model.scratch.parameters(), 'lr' : self.cfg.lr_scratch},
              ])
        elif self.cfg.optimizer.name == 'SGD':
            self.optimizer = SGD([
              {'params' : self.model.pretrained.parameters(), 'lr' : self.cfg.lr_pretrained},
              {'params' : self.model.scratch.parameters(), 'lr' : self.cfg.lr_scratch},
              ])

        # LR scheduler
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=self.cfg.validation.main_val_metric_goal,
            factor=self.cfg.lr_scheduler.factor,
            patience=self.cfg.lr_scheduler.patience,
            min_lr=self.cfg.lr_scheduler.min_lr,
            )

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in self.cfg.eval.eval_metrics]
        # main metric for best checkpoint saving
        self.main_val_metric = self.cfg.validation.main_val_metric
        self.main_val_metric_goal = self.cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in self.cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "min" == self.main_val_metric_goal else -1e8
        self.best_model = copy.deepcopy(self.model.to(self.device).state_dict())
        self.model_datas = {
            'train' : pd.DataFrame({"epoch": []}),  
            'val'   : pd.DataFrame({"epoch": []}),
            }
        self.model_temp_data = pd.DataFrame({})
        self.metric_monitors = {
            'train' : metric.MetricMonitor(),
            'val'   : metric.MetricMonitor(),
        }        
        # Settings
        self.epochs_num = self.cfg.epochs_num
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period


        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming
    
    def train_and_validate(self, t_end=None):
        logging.info("Start training")
        train_start = datetime.now()
        
        device = self.device
        self.model.to(device)
        
        if self.in_evaluation: 
            logging.info("Last evaluation was not finished, will do evaluation before continue training.")
            self.epoch -= 1
            
        for epoch in range(self.epoch, self.epochs_num + 1):
            self.epoch = epoch
            logging.info(f'Epoch [{self.epoch}/{self.epochs_num}]')
            
            for phase in ['train', 'val']:
                if self.in_evaluation and phase=='train':
                    self.in_evaluation = False
                    continue
                
                self.model_datas[phase].at[self.epoch-1, 'epoch'] = self.epoch
                
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                stream = tqdm(skip_first_batches(self.dataloaders[phase], self.n_batch_in_epoch if phase == 'train' else 0))
                for batch in stream:
                    
                    images = batch['rgb_img'].to(device, non_blocking=True)
                    
                    target_for_alig = batch['depth_raw_linear'].numpy()
                    target = batch[self.cfg.gt_depth_type].to(device, non_blocking=True)
                    
                    mask_for_alig = batch['valid_mask_raw'].numpy()
                    mask = batch[self.cfg.gt_mask_type].to(device, non_blocking=True)
                    
                    self.batch_size = images.shape[0]
                    
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase=='train'):
                        output = self.model(images)
                        
                        loss = self.loss(output, target, mask)
                        self.metric_monitors[phase].update("loss", loss.item(), self.batch_size)

                        output_alig = align_depth_least_square(
                            gt_arr=target_for_alig,
                            pred_arr=output.detach().clone().cpu().numpy(),
                            valid_mask_arr=mask_for_alig,
                            return_scale_shift=False,
                            max_resolution=self.cfg.eval.align_max_res,
                            )
                        
                        # Clip to dataset min max
                        if type(self.dataloaders[phase]).__name__ == 'ConcatDataset':
                          output_alig = np.clip(
                              output_alig,
                              a_min=self.dataloaders[phase].dataset.datasets[0].min_depth,
                              a_max=self.dataloaders[phase].dataset.datasets[0].max_depth,
                          )
                        else:
                          output_alig = np.clip(
                              output_alig,
                              a_min=self.dataloaders[phase].dataset.min_depth,
                              a_max=self.dataloaders[phase].dataset.max_depth,
                          )
                        # clip to d > 0 for evaluation
                        output_alig = np.clip(output_alig, a_min=1e-6, a_max=None)
                        
                        # Evaluate
                        sample_metric = []
                        output_alig = torch.from_numpy(output_alig).to(device)
                        for met_func in self.metric_funcs:
                            _metric_name = met_func.__name__
                            _metric = met_func(output_alig, torch.from_numpy(target_for_alig).to(device), mask).item()
                            sample_metric.append(_metric.__str__())
                            self.metric_monitors[phase].update(_metric_name, _metric, self.batch_size)
                        
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.n_batch_in_epoch += 1
                            self.effective_iter += 1
                            if self.effective_iter%len(self.dataloaders[phase]) == 0:
                                logging.debug(
                                    f"iter {self.effective_iter:5d} epoch [{epoch:2d}/{self.epochs_num:2d}]: loss={self.metric_monitors[phase].metrics['loss']['avg']:.5f}"
                                    )
                                logging.debug(
                                    f"lr {self.lr_scheduler.get_last_lr()}, n_batch_in_epoch ({self.n_batch_in_epoch}/{len(self.dataloaders[phase])})"
                                    )                            
                    
                    _is_latest_saved = False        
                    if (
                        self.save_period > 0
                        and 0 == self.effective_iter % self.save_period
                        and not _is_latest_saved
                    ):
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        _is_latest_saved = True
                            
                    if t_end is not None and datetime.now() >= t_end:
                        if not _is_latest_saved:
                            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        time_elapsed = (datetime.now() - train_start).total_seconds()
                        logging.info(f'Time is up, training paused. Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        return
                    
                    torch.cuda.empty_cache()
                    
                if phase == 'train':
                    self.in_evaluation = True
                else:
                    self.lr_scheduler.step(self.metric_monitors[phase].metrics['loss']['avg']) 
                    self.in_evaluation = False
                    
                for metric_name in self.metric_monitors[phase].metrics:
                    self.model_datas[phase].at[self.epoch-1, metric_name] = self.metric_monitors[phase].metrics[metric_name]["avg"]
                self.model_datas[phase].to_csv(os.path.join(self.out_dir_dic['rec'], phase+'_record.csv'), index=False)
                self.metric_monitors[phase].reset()
                
                epoch_main_metric = self.metric_monitors[phase].metrics[self.main_val_metric]['avg']            
                if (
                    phase == 'val'
                    and 
                        (
                            ("min" == self.main_val_metric_goal and epoch_main_metric < self.best_metric)
                            or
                            ("max" == self.main_val_metric_goal and epoch_main_metric > self.best_metric)
                        )
                    ):
                    self.best_metric = epoch_main_metric
                    self.best_model = copy.deepcopy(self.model.state_dict())
                
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)               
                
            self.n_batch_in_epoch = 0
            
        time_elapsed = (datetime.now() - train_start).total_seconds()      
        logging.info(f'Training ended. Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_dic['ckpt'], ckpt_name)
        logging.debug(f"at iteration {self.effective_iter} Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")
        
        os.makedirs(ckpt_dir)

        # Save UNet
        net_path = os.path.join(ckpt_dir, 'net.pth')
        torch.save(self.model.state_dict(), net_path)
        torch.save(self.best_model, os.path.join(ckpt_dir, 'best_net.pth'))
        logging.debug(f"Network weights are saved to: {net_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.debug(f"Trainer state is saved to: {train_state_path}")        

        for metric_name in self.metric_monitors['train'].metrics:
                self.model_temp_data.at[0, metric_name] = self.metric_monitors['train'].metrics[metric_name]["val"]
                self.model_temp_data.at[1, metric_name] = self.metric_monitors['train'].metrics[metric_name]["count"]
                self.model_temp_data.at[2, metric_name] = self.metric_monitors['train'].metrics[metric_name]["avg"]
        self.model_temp_data.to_csv(os.path.join(self.out_dir_dic['rec'],'temp_record.csv'), index=False)
            
        
        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load Net
        _model_path = os.path.join(ckpt_path, "net.pth")
        self.best_model = torch.load(os.path.join(ckpt_path, "best_net.pth"), map_location=self.device)
        self.model.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.to(self.device)
        logging.info(f"Net parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        self.metric_monitors['train'].load(os.path.join(self.out_dir_dic['rec'],'temp_record.csv'))
        
        if os.path.exists(os.path.join(self.out_dir_dic['rec'],'train_record.csv')):
          self.model_datas['train'] = pd.read_csv(os.path.join(self.out_dir_dic['rec'],'train_record.csv'))
        if os.path.exists(os.path.join(self.out_dir_dic['rec'],'eval_record.csv')):
            self.model_datas['val'] = pd.read_csv(os.path.join(self.out_dir_dic['rec'],'eval_record.csv'))
        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
