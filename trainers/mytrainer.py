# Adopted: https://github.com/prs-eth/Marigold/blob/main/src/trainer/marigold_trainer.py

import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from models.MiDas import MidasNet
from models.TernausNet import UNet16

from util import metric
from util.data_loader import skip_first_batches
from util.logging_util import eval_dic_to_text
from util.loss import get_loss
from util.lr_scheduler import IterExponential
from util.alignment import align_depth_least_square, depth2disparity
from util.seeding import generate_seed_sequence

class NetTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        out_dir_tr,
        out_dir_eval,
        val_dataloader: DataLoader,
        test_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: Union[UNet16, MidasNet] = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_tr = out_dir_tr
        self.out_dir_eval = out_dir_eval
        self.train_loader: DataLoader = train_dataloader
        self.val_loader: DataLoader = val_dataloader

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
        # lr_func = IterExponential(
        #     total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
        #     final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
        #     warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        # )
        self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, factor=0.25, patience=50)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name)

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8
        
        self.model_data_train = pd.DataFrame({"epoch": []})
        self.model_data_val = pd.DataFrame({"epoch": []})
        self.model_temp_data = pd.DataFrame({})
        self.metric_monitor_tr = metric.MetricMonitor()
        self.metric_monitor_vl = metric.MetricMonitor()
        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period


        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def train(self, t_end=None):
        logging.info("Start training")
        train_start = datetime.now()
        
        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            self.model_data_train.at[self.epoch, 'epoch'] = self.epoch
            # Skip previous batches when resume
            for batch in tqdm(skip_first_batches(self.train_loader, self.n_batch_in_epoch), position=0, leave=True):
                self.model.train()

                # globally consistent random generators
                if self.seed is None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # Get data
                rgb = batch["rgb_img"].to(device)
                depth_gt = batch[self.gt_depth_type].to(device)
                depth_raw = batch['depth_raw_linear'].numpy()
                depth_raw_met = batch['depth_raw_linear'].to(device)

                if self.gt_mask_type is not None:
                    valid_mask_raw = batch[self.gt_mask_type].numpy()
                    valid_mask = batch[self.gt_mask_type].to(device)
                else:
                    raise NotImplementedError
                

                self.batch_size = rgb.shape[0]

                self.optimizer.zero_grad()

                # Prediction
                model_pred = self.model(rgb)
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")
                                    
                # Masked loss
                batch_loss = self.loss(
                      model_pred.float(),
                      depth_gt.float(),
                      valid_mask
                  )

                loss = batch_loss.mean()
                
                depth_pred: np.ndarray = model_pred.detach().cpu().numpy()
                
                if "least_square" == self.cfg.eval.alignment:
                    depth_pred = align_depth_least_square(
                        gt_arr=depth_raw,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_mask_raw,
                        return_scale_shift=False,
                        max_resolution=self.cfg.eval.align_max_res,
                    )              
                
                # Clip to dataset min max
                if type(self.train_loader.dataset).__name__ == 'ConcatDataset':
                  depth_pred = np.clip(
                      depth_pred,
                      a_min=self.train_loader.dataset.datasets[0].min_depth,
                      a_max=self.train_loader.dataset.datasets[0].max_depth,
                  )                
                else:
                  depth_pred = np.clip(
                      depth_pred,
                      a_min=self.train_loader.dataset.min_depth,
                      a_max=self.train_loader.dataset.max_depth,
                  )
                # clip to d > 0 for evaluation
                depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
                # Evaluate
                sample_metric = []
                depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)
                for met_func in self.metric_funcs:
                    _metric_name = met_func.__name__
                    _metric = met_func(depth_pred_ts, depth_raw_met, valid_mask).item()
                    sample_metric.append(_metric.__str__())
                    self.metric_monitor_tr.update(_metric_name, _metric, self.batch_size)                
                
                self.metric_monitor_tr.update("loss", loss.item(), self.batch_size)
                
                loss.backward()

                self.n_batch_in_epoch += 1

                # Perform optimization step
                self.optimizer.step()

                self.effective_iter += 1

                accumulated_loss = self.metric_monitor_tr.metrics['loss']['avg']

                logging.debug(
                    f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                )
                logging.debug(
                    f"train/{k['avg']}: {v['avg']}" for k, v in self.metric_monitor_tr.metrics
                )
                logging.debug(
                    f"lr {self.lr_scheduler.get_last_lr()}, n_batch_in_epoch {self.n_batch_in_epoch}"
                )

                # Per-step callback
                self._train_step_callback()

                # End of training
                if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(),
                        save_train_state=False,
                    )
                    time_elapsed = (datetime.now() - train_start).total_seconds()      
                    logging.info(f'Training ended. Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    return
                # Time's up
                elif t_end is not None and datetime.now() >= t_end:
                    self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                    time_elapsed = (datetime.now() - train_start).total_seconds()
                    logging.info(f'Time is up, training paused. Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    return

                torch.cuda.empty_cache()
                # <<< Effective batch end <<<
            
            for metric_name in self.metric_monitor_tr.metrics:
                self.model_data_train.at[self.epoch, metric_name] = self.metric_monitor_tr.metrics[metric_name]["avg"]
            self.model_data_train.to_csv(os.path.join(self.out_dir_tr,'train_record.csv'), index=False)
            self.metric_monitor_tr.reset()
            # Validation
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

            # Epoch end
            self.lr_scheduler.step(accumulated_loss)
            self.n_batch_in_epoch = 0
            
        time_elapsed = (datetime.now() - train_start).total_seconds()      
        logging.info(f'Training ended. Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

    @torch.no_grad()
    def validate(self):
        val_dataset_name = self.val_loader.dataset.disp_name
        
        self.model.to(self.device)
        self.metric_monitor_vl.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(self.val_loader))

        for i, batch in enumerate(
            tqdm(self.val_loader, desc=f"evaluating on {self.val_loader.dataset.disp_name}"),
            start=1,
        ):
            # assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_img"].to(self.device)  # .squeeze() [3, H, W]
            # GT depth
            depth_gt = batch[self.gt_depth_type].to(self.device)      
            depth_raw_a = batch['depth_raw_linear'].numpy()      
            depth_raw = batch['depth_raw_linear'].to(self.device) # .squeeze()
            valid_mask_ts = batch[self.gt_mask_type] # .squeeze()
            valid_mask_a = valid_mask_ts.numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth
            model_pred = self.model(rgb_int)

            # Masked loss
            batch_loss = self.loss(
                  model_pred.float(),
                  depth_gt.float(),
                  valid_mask_ts
              )
            loss = batch_loss.mean()

            depth_pred: np.ndarray = model_pred.detach().cpu().numpy()

            if "least_square" == self.cfg.eval.alignment:
                depth_pred = align_depth_least_square(
                    gt_arr=depth_raw_a,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask_a,
                    return_scale_shift=False,
                    max_resolution=self.cfg.eval.align_max_res,
                )

            # Clip to dataset min max
            depth_pred = np.clip(
                depth_pred,
                a_min=self.val_loader.dataset.min_depth,
                a_max=self.val_loader.dataset.max_depth,
            )

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                self.metric_monitor_vl.update(_metric_name, _metric, rgb_int.shape[0])
            self.metric_monitor_vl.update("loss", loss.item(), rgb_int.shape[0])       
        
        logging.info(
            f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {self.metric_monitor_vl}"
        )
        # save to file
        self.model_data_val.at[self.epoch, 'epoch'] = self.epoch
        for metric_name in self.metric_monitor_vl.metrics:
            self.model_data_val.at[self.epoch, metric_name] = self.metric_monitor_vl.metrics[metric_name]["avg"]
        self.model_data_val.to_csv(os.path.join(self.out_dir_eval,f'eval_record.csv'), index=False)

        # Update main eval metric
        main_eval_metric = self.metric_monitor_vl.metrics[self.main_val_metric]["avg"]
        if (
            "minimize" == self.main_val_metric_goal
            and main_eval_metric < self.best_metric
            or "maximize" == self.main_val_metric_goal
            and main_eval_metric > self.best_metric
        ):
            self.best_metric = main_eval_metric
            logging.info(
                f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
            )
            # Save a checkpoint
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.debug(f"Saving checkpoint to: {ckpt_dir}")
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

        for metric_name in self.metric_monitor_tr.metrics:
                self.model_temp_data.at[0, metric_name] = self.metric_monitor_tr.metrics[metric_name]["val"]
                self.model_temp_data.at[1, metric_name] = self.metric_monitor_tr.metrics[metric_name]["count"]
                self.model_temp_data.at[2, metric_name] = self.metric_monitor_tr.metrics[metric_name]["avg"]
        self.model_temp_data.to_csv(os.path.join(self.out_dir_tr,'temp_record.csv'), index=False)
            
        
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

        self.metric_monitor_tr.load(os.path.join(self.out_dir_tr,'temp_record.csv'))
        self.model_data_train = pd.read_csv(os.path.join(self.out_dir_tr,'train_record.csv'))
        if os.path.exists(os.path.join(self.out_dir_eval,'eval_record.csv')):
            self.model_data_val = pd.read_csv(os.path.join(self.out_dir_eval,'eval_record.csv'))
        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"