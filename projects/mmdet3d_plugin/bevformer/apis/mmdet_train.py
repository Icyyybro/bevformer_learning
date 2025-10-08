# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets import custom_build_dataset

def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    """自定义检测器训练函数。
    
    这个函数是BEVFormer项目的核心训练函数，支持分布式训练、验证和自定义评估模型。
    
    Args:
        model: 要训练的模型实例
        dataset: 训练数据集，可以是单个数据集或数据集列表
        cfg: 配置对象，包含所有训练相关的配置参数
        distributed (bool): 是否使用分布式训练。默认为False
        validate (bool): 是否在训练过程中进行验证。默认为False
        timestamp (str, optional): 时间戳，用于日志和检查点文件命名。默认为None
        eval_model (optional): 用于评估的独立模型实例。默认为None
        meta (dict, optional): 包含额外元信息的字典。默认为None
    """
    # 获取根日志记录器
    logger = get_root_logger(cfg.log_level)

    # 准备数据加载器
    # 确保数据集是列表格式，如果是单个数据集则转换为列表
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    
    # 处理已弃用的配置参数 'imgs_per_gpu'
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            # 如果同时存在两个参数，使用 imgs_per_gpu 并发出警告
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            # 如果只有 imgs_per_gpu，自动设置 samples_per_gpu
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # 为每个数据集构建数据加载器
    data_loaders = [
        build_dataloader(
            ds,                                    # 数据集
            cfg.data.samples_per_gpu,             # 每个GPU的样本数
            cfg.data.workers_per_gpu,             # 每个GPU的工作进程数
            len(cfg.gpu_ids),                     # GPU数量（在分布式训练中会被忽略）
            dist=distributed,                     # 是否分布式训练
            seed=cfg.seed,                        # 随机种子
            shuffler_sampler=cfg.data.shuffler_sampler,      # 洗牌采样器配置
            nonshuffler_sampler=cfg.data.nonshuffler_sampler, # 非洗牌采样器配置
        ) for ds in dataset
    ]

    # 将模型放置到GPU上
    if distributed:
        # 分布式训练模式
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # 设置torch.nn.parallel.DistributedDataParallel的find_unused_parameters参数
        model = MMDistributedDataParallel(
            model.cuda(),                                    # 将模型移到CUDA设备
            device_ids=[torch.cuda.current_device()],       # 设备ID列表
            broadcast_buffers=False,                         # 不广播缓冲区
            find_unused_parameters=find_unused_parameters)   # 是否查找未使用的参数
        
        # 如果存在评估模型，也进行相同的分布式包装
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        # 单机多GPU训练模式
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # 构建优化器
    optimizer = build_optimizer(model, cfg.optimizer)

    # 构建运行器（Runner）
    if 'runner' not in cfg:
        # 如果配置中没有runner部分，使用默认配置
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        # 检查total_epochs与runner.max_epochs的一致性
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    
    # 根据是否有评估模型来构建不同的运行器
    if eval_model is not None:
        # 包含评估模型的运行器
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        # 标准运行器
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # 设置运行器的时间戳，确保日志文件名的一致性
    runner.timestamp = timestamp

    # 配置混合精度训练（FP16）
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        # 使用FP16优化器钩子
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        # 分布式训练但没有指定优化器类型时，使用标准优化器钩子
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        # 使用配置中指定的优化器配置
        optimizer_config = cfg.optimizer_config

    # 注册训练相关的钩子
    runner.register_training_hooks(
        cfg.lr_config,                           # 学习率调度配置
        optimizer_config,                        # 优化器配置
        cfg.checkpoint_config,                   # 检查点保存配置
        cfg.log_config,                          # 日志配置
        cfg.get('momentum_config', None))        # 动量配置（可选）
    
    # 注册性能分析钩子（当前被注释掉）
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    # 在分布式训练中注册分布式采样器种子钩子
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # 注册评估钩子
    if validate:
        # 支持验证时batch_size > 1
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # 当前不支持验证时batch_size > 1
            assert False
            # 将'ImageToTensor'替换为'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        
        # 构建验证数据集
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        # 构建验证数据加载器
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,                                    # 验证时不洗牌
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        
        # 配置评估参数
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'  # 根据运行器类型设置评估方式
        # 设置JSON结果文件的保存路径，包含时间戳
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        
        # 根据是否分布式选择相应的评估钩子
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # 注册用户自定义钩子
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        # 验证custom_hooks是列表类型
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        
        # 遍历并注册每个自定义钩子
        for hook_cfg in cfg.custom_hooks:
            # 验证每个钩子配置是字典类型
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            # 获取钩子优先级，默认为'NORMAL'
            priority = hook_cfg.pop('priority', 'NORMAL')
            # 构建并注册钩子
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # 处理模型恢复和加载
    if cfg.resume_from:
        # 从检查点恢复训练（包括优化器状态、学习率调度器状态等）
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # 仅加载模型权重（不包括训练状态）
        runner.load_checkpoint(cfg.load_from)
    
    # 开始训练
    runner.run(data_loaders, cfg.workflow)

