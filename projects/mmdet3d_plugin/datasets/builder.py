
# Copyright (c) OpenMMLab. All rights reserved.
# 导入必要的库
import copy
import platform
import random
from functools import partial

import numpy as np
# 导入MMCV的并行处理和数据整理模块
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
# 导入PyTorch的数据加载器
from torch.utils.data import DataLoader

# 导入MMDetection的采样器
from mmdet.datasets.samplers import GroupSampler
# 导入自定义的分布式采样器
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import DistributedGroupSampler
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import DistributedSampler
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **kwargs):
    """构建PyTorch数据加载器。
    
    在分布式训练中，每个GPU/进程都有一个数据加载器。
    在非分布式训练中，所有GPU共享一个数据加载器。
    
    Args:
        dataset (Dataset): PyTorch数据集对象
        samples_per_gpu (int): 每个GPU上的训练样本数，即每个GPU的批次大小
        workers_per_gpu (int): 每个GPU用于数据加载的子进程数
        num_gpus (int): GPU数量。仅在非分布式训练中使用。默认为1
        dist (bool): 是否进行分布式训练/测试。默认为True
        shuffle (bool): 是否在每个epoch打乱数据。默认为True
        seed (int, optional): 随机种子。默认为None
        shuffler_sampler (dict, optional): 洗牌采样器配置。默认为None
        nonshuffler_sampler (dict, optional): 非洗牌采样器配置。默认为None
        **kwargs: 用于初始化DataLoader的任何关键字参数
        
    Returns:
        DataLoader: PyTorch数据加载器对象
    """
    # 获取分布式训练信息：当前进程的rank和总进程数
    rank, world_size = get_dist_info()
    
    if dist:
        # 分布式训练模式
        # DistributedGroupSampler会确保打乱数据，以满足每个GPU上的图像属于同一组的要求
        if shuffle:
            # 需要打乱数据时，使用分布式组采样器
            sampler = build_sampler(
                shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                dict(
                    dataset=dataset,                # 数据集
                    samples_per_gpu=samples_per_gpu, # 每个GPU的样本数
                    num_replicas=world_size,        # 总进程数
                    rank=rank,                      # 当前进程rank
                    seed=seed)                      # 随机种子
            )
        else:
            # 不需要打乱数据时，使用分布式采样器
            sampler = build_sampler(
                nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                dict(
                    dataset=dataset,                # 数据集
                    num_replicas=world_size,        # 总进程数
                    rank=rank,                      # 当前进程rank
                    shuffle=shuffle,                # 是否打乱
                    seed=seed)                      # 随机种子
            )

        # 分布式训练中，每个GPU的批次大小就是samples_per_gpu
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # 非分布式训练模式
        # 发出警告：在BEVFormer中，非分布式模式仅用于获取推理速度
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        # 如果需要打乱数据，使用组采样器；否则不使用采样器
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        # 非分布式训练中，总批次大小是所有GPU的样本数之和
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    # 如果提供了随机种子，创建工作进程初始化函数
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    # 创建PyTorch数据加载器
    data_loader = DataLoader(
        dataset,                                                    # 数据集
        batch_size=batch_size,                                     # 批次大小
        sampler=sampler,                                           # 采样器
        num_workers=num_workers,                                   # 工作进程数
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu), # 数据整理函数
        pin_memory=False,                                          # 不使用锁页内存
        worker_init_fn=init_fn,                                    # 工作进程初始化函数
        persistent_workers=(num_workers > 0),                      # 是否使用持久化工作进程
        **kwargs)                                                  # 其他参数

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """工作进程初始化函数。
    
    为每个数据加载工作进程设置不同的随机种子，确保数据加载的随机性。
    
    Args:
        worker_id (int): 工作进程ID
        num_workers (int): 总工作进程数
        rank (int): 当前进程的rank
        seed (int): 基础随机种子
    """
    # 每个工作进程的种子 = 工作进程数 * rank + 工作进程ID + 用户种子
    # 这样可以确保每个工作进程都有唯一的随机种子
    worker_seed = num_workers * rank + worker_id + seed
    # 设置numpy和Python的随机种子
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

# 导入MMDetection的数据集注册表
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

# 在非Windows系统上调整文件描述符限制
if platform.system() != 'Windows':
    # 解决PyTorch的文件描述符限制问题
    # 参考：https://github.com/pytorch/pytorch/issues/973
    import resource
    # 获取当前文件描述符限制
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]  # 软限制
    hard_limit = rlimit[1]       # 硬限制
    # 设置新的软限制：至少4096，但不超过硬限制
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

# 创建对象采样器注册表
OBJECTSAMPLERS = Registry('Object sampler')


def custom_build_dataset(cfg, default_args=None):
    """自定义数据集构建函数。
    
    根据配置构建各种类型的数据集，支持数据集包装器和组合。
    
    Args:
        cfg (dict or list): 数据集配置，可以是单个配置字典或配置列表
        default_args (dict, optional): 默认参数。默认为None
        
    Returns:
        Dataset: 构建好的数据集对象
    """
    # 导入各种数据集包装器
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    
    if isinstance(cfg, (list, tuple)):
        # 如果配置是列表或元组，递归构建每个数据集并连接
        dataset = ConcatDataset([custom_build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        # 连接数据集：将多个数据集合并为一个
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))  # 是否分别评估每个子数据集
    elif cfg['type'] == 'RepeatDataset':
        # 重复数据集：将一个数据集重复多次
        dataset = RepeatDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        # 类别平衡数据集：对少数类进行过采样以平衡类别分布
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        # CBGS数据集：类别平衡分组采样数据集
        dataset = CBGSDataset(custom_build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        # 如果注释文件是列表，使用连接数据集函数
        dataset = _concat_dataset(cfg, default_args)
    else:
        # 标准情况：根据配置和注册表构建数据集
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
