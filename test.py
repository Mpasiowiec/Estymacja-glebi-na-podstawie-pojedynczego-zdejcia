import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from models.MiDas import MidasNet
from models.TernausNet import UNet16 

from datasets import BaseDepthDataset, DatasetMode, get_dataset
from datasets.mixed_sampler import MixedBatchSampler
from trainers.tester import NetTester
from util.config_util import recursive_load_config

# from util.depth_transform import (
#     DepthNormalizerBase,
#     get_depth_normalizer,
# )
from util.logging_util import config_logging

if "__main__" == __name__:

    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_nets.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/magisterka",
        help="directory to save checkpoints"
    )
    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        help="Do not use cuda."
    )
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default='/content/drive/MyDrive/magisterka/dane',
        help="directory of training data"
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_false",
        help="Add datetime to the output folder name",
    )
    parser.add_argument(
        "--best_net",
        action="store_false",
        help="best or last to test",
    )
    
    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )

    # -------------------- Initialization --------------------
    # Resume previous run
    if resume_run is not None:
        print(f"Resume run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))

    # Other directories
    out_dir_dic = {
        'ckpt'  : os.path.join(out_dir_run, "checkpoint"),
        'rec' : os.path.join(out_dir_run, "records"),
        'img' : os.path.join(out_dir_run, "img"),
    }
    for key in out_dir_dic.keys():
        if not os.path.exists(out_dir_dic[key]):
            os.makedirs(out_dir_dic[key])
            if key == "img":
                os.makedirs(os.path.join(out_dir_dic[key], "nyu_v2"))
                os.makedirs(os.path.join(out_dir_dic[key], "hypersim"))
                os.makedirs(os.path.join(out_dir_dic[key], "kitti"))
                os.makedirs(os.path.join(out_dir_dic[key], "vkitti2"))

    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")
    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    # depth_transform: DepthNormalizerBase = get_depth_normalizer(
    #     cfg_normalizer=cfg.depth_normalization
    # )
    train_dataset: BaseDepthDataset = get_dataset(
        cfg.dataset.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation_args,
        # depth_transform=depth_transform,
        gt_depth_type=cfg.gt_depth_type
    )
    
    if "mixed" == cfg.dataset.train.name:
        dataset_ls = train_dataset
        assert len(cfg.dataset.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.train_batch_size,
            drop_last=True,
            prob=cfg.dataset.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory, 
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.train_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            generator=loader_generator,
            pin_memory=cfg.dataloader.pin_memory,
        )
    # Validation dataset
    val_dataset: BaseDepthDataset = get_dataset(
        cfg.dataset.val,
        base_data_dir=base_data_dir,
        mode=DatasetMode.EVAL,
        # depth_transform=depth_transform,
        gt_depth_type=cfg.gt_depth_type
    )
    if "mixed" == cfg.dataset.val.name:
        dataset_ls = val_dataset
        assert len(cfg.dataset.val.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.val_batch_size,
            drop_last=True,
            prob=cfg.dataset.val.prob_ls,
            shuffle=False,
            generator=loader_generator,
        )
        val_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
        )
    else:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg.dataloader.val_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=False,
            generator=loader_generator,
            pin_memory=cfg.dataloader.pin_memory,
        )
    # Test dataset
    test_loaders: List[DataLoader] = []
    for _test_dic in cfg.dataset.test:
        _test_dataset = get_dataset(
            _test_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
            # depth_transform=depth_transform,
            gt_depth_type=cfg.gt_depth_type
        )
        _test_loader = DataLoader(
            dataset=_test_dataset,
            batch_size=cfg.dataloader.test_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
        )
        test_loaders.append(_test_loader)

    dataloaders = {
    'train' : train_loader,
    'val' : val_loader,
    'tests' : test_loaders,
    }
    
    # -------------------- Model --------------------
    if cfg.model.name == 'TernausNet':
        model = UNet16(pretrained=True, is_deconv=True)
    elif cfg.model.name == 'MiDas':
      model = MidasNet(backbone=cfg.model.backbone)
    else:
      raise NotImplementedError 

    # -------------------- Trainer --------------------
    logging.debug(f"Trainer: treiner_nets")
    trainer = NetTester(
        cfg=cfg,
        model=model,
        dataloaders=dataloaders,
        device=device,
        out_dir_dic=out_dir_dic,
        best_net=args.best_net
    )
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- test Loop --------------------
    try:
        trainer.test()
    except Exception as e:
        logging.exception(e)
