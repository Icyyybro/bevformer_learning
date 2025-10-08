# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model  # 注释掉原始的训练函数，使用自定义的训练函数

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version


def parse_args():
    """解析命令行参数的函数
    
    Returns:
        args: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description='Train a detector')  # 创建参数解析器
    parser.add_argument('config', help='train config file path')  # 训练配置文件路径
    parser.add_argument('--work-dir', help='the dir to save logs and models')  # 保存日志和模型的目录
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')  # 从指定检查点恢复训练
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')  # 训练期间是否不进行验证
    
    # 创建互斥的GPU参数组（只能选择其中一个）
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')  # GPU数量（仅适用于非分布式训练）
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')  # 指定GPU ID列表
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')  # 随机种子
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')  # 是否设置CUDNN确定性选项
    
    # 已弃用的选项参数
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    
    # 配置选项覆盖参数
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    # 分布式训练启动器选择
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    parser.add_argument('--local_rank', type=int, default=0)  # 本地进程排名
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')  # 是否根据GPU数量自动缩放学习率
    
    args = parser.parse_args()
    
    # 设置本地排名环境变量
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # 检查参数冲突
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    """主函数：执行模型训练的完整流程"""
    # 解析命令行参数
    args = parse_args()

    # 从配置文件加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)  # 合并命令行配置选项
    
    # 从字符串列表导入自定义模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # 从插件目录导入模块，更新注册表
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                # 如果指定了插件目录
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # 使用配置文件所在目录作为导入目录
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            # 导入自定义训练函数
            from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model
    
    # 设置CUDNN基准测试
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # 设置TF32精度（如果需要关闭）
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # 确定工作目录的优先级：命令行 > 配置文件 > 默认文件名
    if args.work_dir is not None:
        # 如果命令行指定了工作目录，则使用命令行参数
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果配置文件中没有工作目录，则使用配置文件名作为默认工作目录
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # 设置恢复训练的检查点文件
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    
    # 设置GPU配置
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    
    # 修复PyTorch 1.8.1中AdamW优化器的bug
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    
    # 根据GPU数量自动缩放学习率（应用线性缩放规则）
    if args.autoscale_lr:
        # 应用线性缩放规则 (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # 初始化分布式环境（必须在logger之前，因为logger依赖分布式信息）
    if args.launcher == 'none':
        distributed = False  # 非分布式训练
    else:
        distributed = True  # 分布式训练
        init_dist(args.launcher, **cfg.dist_params)
        # 在分布式训练模式下重新设置gpu_ids
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 将配置文件转储到工作目录
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # 在其他步骤之前初始化logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    
    # 指定logger名称，如果仍使用'mmdet'，输出信息将被过滤且不会保存到日志文件
    # TODO: 判断是否训练检测或分割模型的临时解决方案
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'  # 分割模型使用mmseg logger
    else:
        logger_name = 'mmdet'  # 检测模型使用mmdet logger
    
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # 初始化元数据字典，记录重要信息如环境信息和种子
    meta = dict()
    
    # 记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # 记录基本信息
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 设置随机种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # 构建模型
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),  # 训练配置
        test_cfg=cfg.get('test_cfg'))    # 测试配置
    model.init_weights()  # 初始化模型权重

    logger.info(f'Model:\n{model}')
    
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]  # 训练数据集
    
    # 如果工作流程包含验证步骤，则构建验证数据集
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # 处理数据集包装器的情况
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # 在深拷贝的配置中设置test_mode=False
        # 这不会影响后续的AP/AR计算
        # 参考：https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    
    # 设置检查点配置的元数据
    if cfg.checkpoint_config is not None:
        # 在检查点中保存mmdet版本、配置文件内容和类名作为元数据
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # 用于分割器
            if hasattr(datasets[0], 'PALETTE') else None)
    
    # 为可视化方便添加类别属性
    model.CLASSES = datasets[0].CLASSES
    
    # 调用自定义训练函数开始训练
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),  # 是否进行验证
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
