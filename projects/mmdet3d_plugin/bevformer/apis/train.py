# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

# 导入自定义的检测器训练函数
from .mmdet_train import custom_train_detector
# 导入分割器训练函数
from mmseg.apis import train_segmentor
# 导入标准检测器训练函数
from mmdet.apis import train_detector

def custom_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                eval_model=None,
                meta=None):
    """根据配置启动模型训练的函数包装器。

    由于我们在runner中需要不同的eval_hook，这个函数提供了自定义的训练逻辑。
    在未来版本中应该被弃用。
    
    Args:
        model: 要训练的模型实例
        dataset: 训练数据集
        cfg: 配置对象，包含训练相关的所有配置
        distributed (bool): 是否使用分布式训练。默认为False
        validate (bool): 是否在训练过程中进行验证。默认为False
        timestamp (str, optional): 时间戳，用于日志和检查点文件命名。默认为None
        eval_model (optional): 用于评估的模型实例，可能与训练模型不同。默认为None
        meta (dict, optional): 包含额外元信息的字典。默认为None
    """
    # 检查模型类型，如果是3D编码解码器模型则抛出错误
    if cfg.model.type in ['EncoderDecoder3D']:
        assert False  # 当前不支持3D编码解码器模型的训练
    else:
        # 对于其他类型的模型（如检测器），使用自定义的检测器训练函数
        custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            eval_model=eval_model,
            meta=meta)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """根据配置启动模型训练的标准函数包装器。

    由于我们在runner中需要不同的eval_hook，这个函数提供了标准的训练逻辑。
    在未来版本中应该被弃用。
    
    Args:
        model: 要训练的模型实例
        dataset: 训练数据集
        cfg: 配置对象，包含训练相关的所有配置
        distributed (bool): 是否使用分布式训练。默认为False
        validate (bool): 是否在训练过程中进行验证。默认为False
        timestamp (str, optional): 时间戳，用于日志和检查点文件命名。默认为None
        meta (dict, optional): 包含额外元信息的字典。默认为None
    """
    # 根据模型类型选择相应的训练函数
    if cfg.model.type in ['EncoderDecoder3D']:
        # 如果是3D编码解码器模型（分割模型），使用分割器训练函数
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        # 如果是其他类型的模型（如检测器），使用标准的检测器训练函数
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
