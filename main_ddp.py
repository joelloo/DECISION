import argparse
import os
import time
from typing import Callable
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from models.inet import INet
from dataset import SeqDataset, DistributedSubsetBatchSampler
from utils import AverageMeter, Timer


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        criterion: Callable,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        sync_bn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = DDP(sync_bn_model, device_ids=[gpu_id], find_unused_parameters=True)
        self.criterion = criterion
        self.scheduler = scheduler

    def _run_batch(self, obs, intentions, labels, running_loss):
        self.optimizer.zero_grad()

        obs = obs.to(self.gpu_id)
        intentions = intentions.to(self.gpu_id)
        labels = labels[:, -1].to(self.gpu_id)
        left, mid, right = torch.split(obs, obs.size(4) // 3, dim=4)
        outs = self.model(left, mid, right, intentions)
        loss = self.criterion(outs, labels)

        loss.backward()
        running_loss.update(loss.item())
        self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch}")
        self.train_data.batch_sampler.set_epoch(epoch)

        # TODO: Do we need to init dataset?
        if isinstance(self.train_data.dataset, SeqDataset):
            print("Init dataset on GPU:", self.gpu_id)
            self.train_data.dataset.init_dataset()

        running_loss = AverageMeter()
        self.model.train()

        pg = tqdm(self.train_data, leave=False, total=len(self.train_data))
        timer = Timer()
        timer.tic()

        for obs, intentions, labels in pg:
            timer.toc("Start batch")
            timer.tic()

            self._run_batch(obs, intentions, labels, running_loss)
            pg.set_postfix({
                'train loss': '{:.6f}'.format(running_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

            timer.toc("End batch")
            timer.tic()

        self.scheduler.step()

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(dataset_path, image_shape, num_frames, frame_interval, 
                    downsample_ratio, num_modes, dropout, batch_size):
    # annotations_path = os.path.join(dataset_path, 'nas_train_v17.txt')
    annotations_path = os.path.join(dataset_path, "train.txt")
    train_set = SeqDataset(annotations_path, dataset_path, image_shape, num_frames,
                           frame_interval, aug=True, keep_prob=downsample_ratio, flip=True,
                           num_intention=num_modes, elevator_only=False)
    model = INet(pretrained=True, fc_dropout_keep=dropout, intent_feat=False, num_modes=3,
                    num_frames=num_frames)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7 * batch_size * num_frames, 
                                  weight_decay=1e-4)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    train_sampler = DistributedSubsetBatchSampler(dataset, batch_size=batch_size)
    return DataLoader(
        dataset,
        pin_memory=True,
        shuffle=False,
        batch_sampler=train_sampler,
        num_workers=4
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, 
         batch_size: int, num_views: int, input_size: int, dataset_path: str,
         num_frames: int, frame_interval: int, downsample_ratio: float,
         num_modes: int, dropout: float):
    print("Setting up on GPU:", rank)
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(
        dataset_path, (input_size, input_size * num_views), num_frames,
        frame_interval, downsample_ratio, num_modes, dropout, batch_size)    
    train_data = prepare_dataloader(dataset, batch_size)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 140], gamma=0.1)
    trainer = Trainer(model, train_data, optimizer, rank, save_every, 
                         torch.nn.MSELoss(), scheduler)
    
    print("Training on GPU:", rank)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model', default=200)
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot', default=40)
    parser.add_argument('--batch_size', default=128, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--num_views', default=3, type=int, help='Number of cameras')
    parser.add_argument('--input_size', type=int, help='the size of input visual percepts', default=112)
    parser.add_argument('--dataset_path', type=str, help='path to train dataset', default='/data/home/joel/storage/inet_data')
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--downsample_ratio', help='the ratio by which to downsample particular samples in the dataset',
                        type=float, default=0.1)
    parser.add_argument('--num_modes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--frame_interval', help='sample 1 frame every x frames', type=int, default=1)
    parser.add_argument('--world_size', help='number of GPUs to use', type=int, default=4)
    args = parser.parse_args()

    world_size = torch.cuda.device_count() if args.world_size is None else args.world_size
    mp.spawn(
        main, 
        args=(
            world_size, 
            args.save_every, 
            args.total_epochs, 
            args.batch_size, 
            args.num_views,
            args.input_size,
            args.dataset_path,
            args.num_frames,
            args.frame_interval,
            args.downsample_ratio,
            args.num_modes,
            args.dropout,
        ), 
        nprocs=world_size
    )