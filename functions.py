# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch

# Other
import os
import glob
import contextlib
import socket
import datetime

def find_last_checkpoint(callback_path, return_full_path=False):

    # All Checkpoints
    checkpoints = glob.glob(os.path.join(callback_path, "checkpoints_*.ckpt"))

    # Select Last Checkpoint else None
    max_steps = 0
    last_checkpoint = None
    for checkpoint in checkpoints:
        checkpoint = checkpoint.split("/")[-1]
        checkpoint_steps = int(checkpoint.split("_")[-1].replace(".ckpt", ""))
        if checkpoint_steps > max_steps:
            max_steps = checkpoint_steps
            last_checkpoint = checkpoint

    # Join path
    if last_checkpoint != None and return_full_path:
        last_checkpoint = os.path.join(callback_path, last_checkpoint)

    return last_checkpoint

def load_model(args):

    # Model Device
    device = torch.device("cuda:" + str(args.rank) if torch.cuda.is_available() and not args.cpu else "cpu")
    if "cuda" in str(device):
        print("Rank {} device: {}, {}, {}MB".format(args.rank, device, torch.cuda.get_device_properties(device).name, int(torch.cuda.get_device_properties(device).total_memory // 1e6)))
        args.num_gpus = torch.cuda.device_count()
    else:
        print("Rank {} device: {}".format(args.rank, device))
        args.num_gpus = 1

    # Barrier
    if args.distributed:
        torch.distributed.barrier()

    # Set Model Device
    model = args.config.model.to(device)

    # Set EMA Model
    if hasattr(args.config, "ema_tau") and args.rank == 0:
        model.set_ema(args.config.ema_tau)

    # Set Callback Path
    args.config.callback_path = getattr(args.config, "callback_path", os.path.join("callbacks", "/".join(args.config_file.replace(".py", "").split("/")[1:])))
    # Append callback Tag
    if hasattr(args.config, "callback_tag"):
        args.config.callback_path = os.path.join(args.config.callback_path, args.config.callback_tag)
    # Append Date Time
    if getattr(args.config, "datetime_callback", args.datetime_callback):
        date_time = str(datetime.datetime.now()).replace(" ", "_").replace(".", "_")
        args.config.callback_path = os.path.join(args.config.callback_path, date_time)
        if args.distributed:
            object_list = [args.config.callback_path]
            torch.distributed.broadcast_object_list(object_list, src=0)
            args.config.callback_path = object_list[0]

    # Last Checkpoint
    if args.load_last:
        last_checkpoint = find_last_checkpoint(args.config.callback_path)
        if last_checkpoint != None:
            args.checkpoint = last_checkpoint

    # Load Checkpoint
    if args.checkpoint is not None:
        model.load(os.path.join(args.config.callback_path, args.checkpoint), strict=not args.no_strict)

    # Barrier
    if args.distributed:
        torch.distributed.barrier()

    # Model Summary
    if args.rank == 0:
        model.summary(show_dict=args.show_dict, show_modules=args.show_modules)

    # Distribute Strategy
    if args.distributed:
        if args.rank == 0:
            print("Parallelize model on", args.world_size, "GPUs")
        model.distribute_strategy(args.rank)

    # Parallel Strategy
    if args.parallel and not args.distributed:
        print("Parallelize model on", torch.cuda.device_count(), "GPUs")
        model.parallel_strategy()

    return model

def load_datasets(args):

    def print_dataset(args, dataset, tag):

        if args.distributed:
            print("{} Dataset: {}, {:,} samples - {:,} batches - batch size {} x {}".format(tag, dataset.dataset.__class__.__name__, len(dataset.dataset), len(dataset), dataset.dataset.batch_size, args.num_gpus))
        else:
            print("{} Dataset: {}, {:,} samples - {:,} batches - batch size {}".format(tag, dataset.dataset.__class__.__name__, len(dataset.dataset), len(dataset), dataset.dataset.batch_size))

    # Training Dataset
    if hasattr(args.config, "training_dataset"):

        # DataLoader
        dataset_train = torch.utils.data.DataLoader(
            dataset=args.config.training_dataset,
            batch_size=args.config.training_dataset.batch_size,
            shuffle=False if args.distributed else args.config.training_dataset.shuffle,
            sampler=torch.utils.data.distributed.DistributedSampler(args.config.training_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.config.training_dataset.shuffle) if (args.distributed and not isinstance(args.config.training_dataset, torch.utils.data.IterableDataset)) else None,
            num_workers=args.config.training_dataset.num_workers,
            collate_fn=args.config.training_dataset.collate_fn,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=getattr(args.config, "worker_init_fn", None),
            persistent_workers=args.config.training_dataset.persistent_workers,
            #prefetch_factor=getattr(args.config, "prefetch_factor", 2) if args.config.training_dataset.num_workers > 0 else None
        )
        
        # Loaded Print
        if args.rank == 0:
            print_dataset(args, dataset_train, "Training")

    else:

        dataset_train = None

    # Evaluation Dataset
    if hasattr(args.config, "evaluation_dataset"):

        # Multiple Evaluation datasets
        if isinstance(args.config.evaluation_dataset, list):

            dataset_eval = []
            for dataset in args.config.evaluation_dataset:

                # DataLoader
                dataset_eval.append(torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=dataset.batch_size,
                    shuffle=False if args.distributed else dataset.shuffle,
                    sampler=torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=dataset.shuffle) if (args.distributed and not isinstance(args.config.evaluation_dataset, torch.utils.data.IterableDataset)) else None,
                    num_workers=dataset.num_workers,
                    collate_fn=dataset.collate_fn,
                    pin_memory=False,
                    drop_last=False,
                    worker_init_fn=getattr(args.config, "worker_init_fn", None),
                    persistent_workers=dataset.persistent_workers,
                    #prefetch_factor=getattr(args.config, "prefetch_factor", 2) if dataset.num_workers > 0 else None
                ))
            
                # Loaded Print
                if args.rank == 0:
                    print_dataset(args, dataset_eval[-1], "Evaluation")

        # One Evaluation dataset
        else:

            # DataLoader
            dataset_eval = torch.utils.data.DataLoader(
                dataset=args.config.evaluation_dataset,
                batch_size=args.config.evaluation_dataset.batch_size,
                shuffle=False if args.distributed else args.config.evaluation_dataset.shuffle,
                sampler=torch.utils.data.distributed.DistributedSampler(args.config.evaluation_dataset, num_replicas=args.world_size,rank=args.rank, shuffle=args.config.evaluation_dataset.shuffle) if (args.distributed and not isinstance(args.config.evaluation_dataset, torch.utils.data.IterableDataset)) else None,
                num_workers=args.config.evaluation_dataset.num_workers,
                collate_fn=args.config.evaluation_dataset.collate_fn,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=getattr(args.config, "worker_init_fn", None),
                persistent_workers=args.config.evaluation_dataset.persistent_workers,
                #prefetch_factor=getattr(args.config, "prefetch_factor", 2) if args.config.evaluation_dataset.num_workers > 0 else None
            )
            
            # Loaded Print
            if args.rank == 0:
                print_dataset(args, dataset_eval, "Evaluation")
                
    else:
        dataset_eval = None
    
    return dataset_train, dataset_eval

def get_open_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]