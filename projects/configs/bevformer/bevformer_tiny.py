# BEVFormer-tiny模型配置文件
# BEVFormer-tiny至少消耗6700M GPU显存
# 相比于bevformer_base，bevformer_tiny具有以下特点：
# 更小的骨干网络: R101-DCN -> R50
# 更小的BEV特征图: 200*200 -> 50*50  
# 更少的编码器层数: 6 -> 3
# 更小的输入尺寸: 1600*900 -> 800*450
# 多尺度特征 -> 单尺度特征 (C5)

# 基础配置文件继承
_base_ = [
    '../datasets/custom_nus-3d.py',    # nuScenes数据集配置
    '../_base_/default_runtime.py'     # 默认运行时配置
]

# 插件配置
plugin = True                          # 启用插件系统
plugin_dir = 'projects/mmdet3d_plugin/' # 插件目录路径

# 点云范围配置 - 定义3D检测的空间范围
# 如果点云范围改变，模型也应该相应地改变其点云范围
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # [x_min, y_min, z_min, x_max, y_max, z_max]
voxel_size = [0.2, 0.2, 8]           # 体素大小 [x, y, z] 单位：米

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],   # RGB通道均值 (ImageNet预训练模型标准)
    std=[58.395, 57.12, 57.375],      # RGB通道标准差
    to_rgb=True                       # 是否转换为RGB格式
)

# nuScenes数据集的10类目标检测类别
class_names = [
    'car',                    # 汽车
    'truck',                  # 卡车
    'construction_vehicle',   # 工程车辆
    'bus',                    # 公交车
    'trailer',                # 拖车
    'barrier',                # 护栏
    'motorcycle',             # 摩托车
    'bicycle',                # 自行车
    'pedestrian',             # 行人
    'traffic_cone'            # 交通锥
]

# 输入模态配置 - 定义使用哪些传感器数据
input_modality = dict(
    use_lidar=False,     # 不使用激光雷达数据
    use_camera=True,     # 使用相机数据
    use_radar=False,     # 不使用雷达数据
    use_map=False,       # 不使用地图数据
    use_external=True    # 使用外部信息（如CAN总线数据）
)

# 模型架构参数定义
_dim_ = 256              # 特征维度
_pos_dim_ = _dim_//2     # 位置编码维度 (128)
_ffn_dim_ = _dim_*2      # 前馈网络维度 (512)
_num_levels_ = 1         # 特征层级数量（tiny版本使用单尺度）
bev_h_ = 50              # BEV特征图高度
bev_w_ = 50              # BEV特征图宽度
queue_length = 3         # 时序队列长度，每个序列包含3帧

# 模型配置
model = dict(
    type='BEVFormer',           # 模型类型
    use_grid_mask=True,         # 使用网格掩码数据增强
    video_test_mode=True,       # 视频测试模式（时序推理）
    pretrained=dict(img='torchvision://resnet50'),  # 图像骨干网络预训练权重
    
    # 图像骨干网络配置
    img_backbone=dict(
        type='ResNet',          # 骨干网络类型：ResNet
        depth=50,               # 网络深度：ResNet-50
        num_stages=4,           # 阶段数量
        out_indices=(3,),       # 输出特征层索引（只使用第4阶段，即C5）
        frozen_stages=1,        # 冻结前1个阶段的参数
        norm_cfg=dict(type='BN', requires_grad=False),  # 批归一化配置，不更新参数
        norm_eval=True,         # 评估模式下的归一化
        style='pytorch'         # PyTorch风格的ResNet
    ),
    
    # 图像颈部网络配置（特征金字塔网络）
    img_neck=dict(
        type='FPN',                    # 特征金字塔网络
        in_channels=[2048],            # 输入通道数（ResNet-50的C5层）
        out_channels=_dim_,            # 输出通道数（256）
        start_level=0,                 # 起始层级
        add_extra_convs='on_output',   # 在输出上添加额外卷积
        num_outs=_num_levels_,         # 输出层数（1层）
        relu_before_extra_convs=True   # 额外卷积前使用ReLU
    ),
    
    # 3D目标检测头配置
    pts_bbox_head=dict(
        type='BEVFormerHead',          # 检测头类型
        bev_h=bev_h_,                  # BEV高度（50）
        bev_w=bev_w_,                  # BEV宽度（50）
        num_query=900,                 # 查询数量
        num_classes=10,                # 类别数量
        in_channels=_dim_,             # 输入通道数（256）
        sync_cls_avg_factor=True,      # 同步分类平均因子
        with_box_refine=True,          # 启用边界框精炼
        as_two_stage=False,            # 不使用两阶段检测
        
        # Transformer配置
        transformer=dict(
            type='PerceptionTransformer',  # 感知Transformer
            rotate_prev_bev=True,          # 旋转前一帧BEV特征
            use_shift=True,                # 使用位移
            use_can_bus=True,              # 使用CAN总线数据
            embed_dims=_dim_,              # 嵌入维度（256）
            
            # 编码器配置
            encoder=dict(
                type='BEVFormerEncoder',   # BEVFormer编码器
                num_layers=3,              # 编码器层数（tiny版本减少到3层）
                pc_range=point_cloud_range, # 点云范围
                num_points_in_pillar=4,    # 每个柱子中的点数
                return_intermediate=False,  # 不返回中间结果
                
                # Transformer层配置
                transformerlayers=dict(
                    type='BEVFormerLayer',     # BEVFormer层
                    attn_cfgs=[                # 注意力配置列表
                        # 时序自注意力
                        dict(
                            type='TemporalSelfAttention',  # 时序自注意力
                            embed_dims=_dim_,              # 嵌入维度
                            num_levels=1                   # 层级数
                        ),
                        # 空间交叉注意力
                        dict(
                            type='SpatialCrossAttention',  # 空间交叉注意力
                            pc_range=point_cloud_range,    # 点云范围
                            deformable_attention=dict(     # 可变形注意力配置
                                type='MSDeformableAttention3D',  # 多尺度可变形注意力3D
                                embed_dims=_dim_,                 # 嵌入维度
                                num_points=8,                     # 采样点数
                                num_levels=_num_levels_           # 特征层级数
                            ),
                            embed_dims=_dim_,              # 嵌入维度
                        )
                    ],
                    feedforward_channels=_ffn_dim_,    # 前馈网络通道数（512）
                    ffn_dropout=0.1,                   # 前馈网络dropout率
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')  # 操作顺序
                )
            ),
            
            # 解码器配置
            decoder=dict(
                type='DetectionTransformerDecoder',  # 检测Transformer解码器
                num_layers=6,                        # 解码器层数
                return_intermediate=True,            # 返回中间结果
                
                # Transformer层配置
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',  # DETR解码器层
                    attn_cfgs=[                          # 注意力配置
                        # 多头自注意力
                        dict(
                            type='MultiheadAttention',   # 多头注意力
                            embed_dims=_dim_,            # 嵌入维度
                            num_heads=8,                 # 注意力头数
                            dropout=0.1                  # dropout率
                        ),
                        # 自定义多尺度可变形注意力
                        dict(
                            type='CustomMSDeformableAttention',  # 自定义多尺度可变形注意力
                            embed_dims=_dim_,                    # 嵌入维度
                            num_levels=1                         # 层级数
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,      # 前馈网络通道数
                    ffn_dropout=0.1,                     # 前馈网络dropout率
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')  # 操作顺序
                )
            )
        ),
        
        # 边界框编码器配置
        bbox_coder=dict(
            type='NMSFreeCoder',           # 无NMS编码器
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 后处理中心范围
            pc_range=point_cloud_range,    # 点云范围
            max_num=300,                   # 最大检测数量
            voxel_size=voxel_size,         # 体素大小
            num_classes=10                 # 类别数量
        ),
        
        # 位置编码配置
        positional_encoding=dict(
            type='LearnedPositionalEncoding',  # 学习的位置编码
            num_feats=_pos_dim_,               # 特征数量（128）
            row_num_embed=bev_h_,              # 行嵌入数量（50）
            col_num_embed=bev_w_,              # 列嵌入数量（50）
        ),
        
        # 损失函数配置
        loss_cls=dict(                    # 分类损失
            type='FocalLoss',             # Focal Loss
            use_sigmoid=True,             # 使用sigmoid激活
            gamma=2.0,                    # gamma参数
            alpha=0.25,                   # alpha参数
            loss_weight=2.0               # 损失权重
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),      # 边界框L1损失
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)       # GIoU损失（权重为0，不使用）
    ),
    
    # 模型训练配置
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],       # 网格大小
        voxel_size=voxel_size,         # 体素大小
        point_cloud_range=point_cloud_range,  # 点云范围
        out_size_factor=4,             # 输出尺寸因子
        assigner=dict(                 # 分配器配置
            type='HungarianAssigner3D',    # 匈牙利分配器3D
            cls_cost=dict(type='FocalLossCost', weight=2.0),    # 分类代价
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),    # 回归代价
            iou_cost=dict(type='IoUCost', weight=0.0),          # IoU代价（虚假代价，为了兼容DETR头）
            pc_range=point_cloud_range     # 点云范围
        )
    ))
)

# 数据集配置
dataset_type = 'CustomNuScenesDataset'  # 自定义nuScenes数据集
data_root = 'data/nuScenes/'            # 数据根目录
file_client_args = dict(backend='disk') # 文件客户端参数

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),           # 加载多视角图像
    dict(type='PhotoMetricDistortionMultiViewImage'),                    # 光度失真数据增强
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),  # 加载3D标注
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range), # 目标范围过滤
    dict(type='ObjectNameFilter', classes=class_names),                  # 目标名称过滤
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),                # 多视角图像归一化
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),           # 随机缩放（缩放到0.5倍）
    dict(type='PadMultiViewImage', size_divisor=32),                     # 填充图像（32的倍数）
    dict(type='DefaultFormatBundle3D', class_names=class_names),         # 默认格式打包
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])  # 收集数据
]

# 测试数据处理流水线
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),           # 加载多视角图像
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),                # 多视角图像归一化
    
    # 多尺度翻转增强
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),         # 图像尺寸
        pts_scale_ratio=1,             # 点云缩放比例
        flip=False,                    # 不翻转
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),   # 随机缩放
            dict(type='PadMultiViewImage', size_divisor=32),             # 填充图像
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False           # 测试时不需要标签
            ),
            dict(type='CustomCollect3D', keys=['img'])                   # 只收集图像
        ]
    )
]

# 数据配置
data = dict(
    samples_per_gpu=1,                 # 每个GPU的样本数
    workers_per_gpu=4,                 # 每个GPU的工作进程数
    
    # 训练数据配置
    train=dict(
        type=dataset_type,             # 数据集类型
        data_root=data_root,           # 数据根目录
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',  # 训练标注文件
        pipeline=train_pipeline,       # 数据处理流水线
        classes=class_names,           # 类别名称
        modality=input_modality,       # 输入模态
        test_mode=False,               # 非测试模式
        use_valid_flag=True,           # 使用有效标志
        bev_size=(bev_h_, bev_w_),     # BEV尺寸
        queue_length=queue_length,     # 队列长度
        box_type_3d='LiDAR'           # 3D边界框类型（LiDAR坐标系）
    ),
    
    # 验证数据配置
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',    # 验证标注文件
        pipeline=test_pipeline,        # 测试流水线
        bev_size=(bev_h_, bev_w_),     # BEV尺寸
        classes=class_names,           # 类别名称
        modality=input_modality,       # 输入模态
        samples_per_gpu=1              # 每个GPU样本数
    ),
    
    # 测试数据配置
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',    # 测试标注文件
        pipeline=test_pipeline,        # 测试流水线
        bev_size=(bev_h_, bev_w_),     # BEV尺寸
        classes=class_names,           # 类别名称
        modality=input_modality        # 输入模态
    ),
    
    # 采样器配置
    shuffler_sampler=dict(type='DistributedGroupSampler'),      # 分布式分组采样器
    nonshuffler_sampler=dict(type='DistributedSampler')        # 分布式采样器
)

# 优化器配置
optimizer = dict(
    type='AdamW',                      # AdamW优化器
    lr=2e-4,                          # 学习率
    paramwise_cfg=dict(               # 参数特定配置
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # 图像骨干网络学习率倍数（0.1倍）
        }
    ),
    weight_decay=0.01                 # 权重衰减
)

# 优化器配置
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 梯度裁剪

# 学习率策略
lr_config = dict(
    policy='CosineAnnealing',         # 余弦退火策略
    warmup='linear',                  # 线性预热
    warmup_iters=500,                 # 预热迭代次数
    warmup_ratio=1.0 / 3,             # 预热比例
    min_lr_ratio=1e-3                 # 最小学习率比例
)

total_epochs = 24                     # 总训练轮数
evaluation = dict(interval=1, pipeline=test_pipeline)  # 评估配置：每轮评估一次

# 运行器配置
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)  # 基于轮次的运行器

# 日志配置
log_config = dict(
    interval=50,                      # 日志记录间隔（每50次迭代）
    hooks=[
        dict(type='TextLoggerHook'),      # 文本日志钩子
        dict(type='TensorboardLoggerHook') # Tensorboard日志钩子
    ]
)

# 检查点配置
checkpoint_config = dict(interval=1)  # 检查点保存间隔（每轮保存一次）
