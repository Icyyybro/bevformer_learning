# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer 3D目标检测器
    
    BEVFormer是一个基于Transformer的多视角3D目标检测模型，通过时空Transformer
    将多视角图像特征转换为鸟瞰图(BEV)表示，实现端到端的3D目标检测。
    
    Args:
        video_test_mode (bool): 是否在推理时使用时序信息。
    """

    def __init__(self,
                 use_grid_mask=False,           # 是否使用GridMask数据增强
                 pts_voxel_layer=None,          # 点云体素化层（BEVFormer中不使用）
                 pts_voxel_encoder=None,        # 点云体素编码器（BEVFormer中不使用）
                 pts_middle_encoder=None,       # 点云中间编码器（BEVFormer中不使用）
                 pts_fusion_layer=None,         # 点云融合层（BEVFormer中不使用）
                 img_backbone=None,             # 图像骨干网络（如ResNet）
                 pts_backbone=None,             # 点云骨干网络（BEVFormer中不使用）
                 img_neck=None,                 # 图像颈部网络（如FPN）
                 pts_neck=None,                 # 点云颈部网络（BEVFormer中不使用）
                 pts_bbox_head=None,            # 3D边界框检测头（BEVFormerHead）
                 img_roi_head=None,             # 图像ROI头（BEVFormer中不使用）
                 img_rpn_head=None,             # 图像RPN头（BEVFormer中不使用）
                 train_cfg=None,                # 训练配置
                 test_cfg=None,                 # 测试配置
                 pretrained=None,               # 预训练模型路径
                 video_test_mode=False          # 视频测试模式，是否使用时序信息
                 ):

        # 调用父类初始化函数
        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        
        # 初始化GridMask数据增强
        # GridMask是一种图像数据增强技术，通过在图像上添加网格状遮罩来提高模型鲁棒性
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # 时序相关配置
        self.video_test_mode = video_test_mode  # 是否在测试时使用时序信息
        # 存储前一帧的信息，用于时序建模
        self.prev_frame_info = {
            'prev_bev': None,      # 前一帧的BEV特征
            'scene_token': None,   # 场景标识符
            'prev_pos': 0,         # 前一帧的位置
            'prev_angle': 0,       # 前一帧的角度
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """从图像中提取特征
        
        Args:
            img (torch.Tensor): 输入图像，形状为 (B, N, C, H, W) 或 (B*N, C, H, W)
                B: batch size, N: 相机数量, C: 通道数, H: 高度, W: 宽度
            img_metas (list[dict]): 图像元信息列表
            len_queue (int, optional): 时序队列长度，用于处理时序数据
            
        Returns:
            list[torch.Tensor]: 多尺度图像特征列表
        """
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # 更新每个图像的真实输入形状
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            # 处理不同维度的输入图像
            if img.dim() == 5 and img.size(0) == 1:
                # 如果batch size为1，去除batch维度
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                # 将5维张量重塑为4维：(B, N, C, H, W) -> (B*N, C, H, W)
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            
            # 应用GridMask数据增强（仅在训练时）
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # 通过骨干网络提取图像特征
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        # 如果有颈部网络（如FPN），进一步处理特征
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # 重新整理特征形状以适应多视角输入
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                # 处理时序数据：(B*len_queue*N, C, H, W) -> (B, len_queue, N, C, H, W)
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                # 处理单帧数据：(B*N, C, H, W) -> (B, N, C, H, W)
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """从图像和点云中提取特征（BEVFormer只使用图像）
        
        Args:
            img (torch.Tensor): 输入图像
            img_metas (list[dict], optional): 图像元信息
            len_queue (int, optional): 时序队列长度
            
        Returns:
            list[torch.Tensor]: 图像特征列表
        """
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,           # 图像特征（在BEVFormer中pts_feats实际是img_feats）
                          gt_bboxes_3d,        # 真实3D边界框
                          gt_labels_3d,        # 真实3D标签
                          img_metas,           # 图像元信息
                          gt_bboxes_ignore=None,  # 要忽略的真实边界框
                          prev_bev=None):      # 前一帧的BEV特征
        """训练时的前向传播函数
        
        Args:
            pts_feats (list[torch.Tensor]): 点云分支特征（在BEVFormer中是图像特征）
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): 每个样本的真实3D边界框
            gt_labels_3d (list[torch.Tensor]): 每个样本的真实3D标签
            img_metas (list[dict]): 样本的元信息
            gt_bboxes_ignore (list[torch.Tensor], optional): 要忽略的真实边界框
            prev_bev (torch.Tensor, optional): 前一帧的BEV特征
            
        Returns:
            dict: 各分支的损失
        """
        # 通过BEVFormerHead进行前向传播，获取预测结果
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        
        # 准备损失计算的输入
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        
        # 计算损失
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        """虚拟前向传播，用于模型分析"""
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """根据return_loss参数调用训练或测试前向传播
        
        注意：这个设置会改变期望的输入格式。当return_loss=True时，
        img和img_metas是单层嵌套的（即torch.Tensor和list[dict]），
        当return_loss=False时，img和img_metas应该是双层嵌套的
        （即list[torch.Tensor], list[list[dict]]），外层列表表示测试时增强。
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
       
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """迭代获取历史BEV特征。为了节省GPU内存，不计算梯度。
        
        Args:
            imgs_queue (torch.Tensor): 历史图像队列，形状为(bs, len_queue, num_cams, C, H, W)
            img_metas_list (list[list[dict]]): 历史图像元信息列表
            
        Returns:
            torch.Tensor: 最后一帧的BEV特征
        """
        # 设置为评估模式
        self.eval()

        with torch.no_grad():  # 不计算梯度以节省内存
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            # 重塑图像队列形状以便批处理
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            # 一次性提取所有帧的图像特征
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            
            # 逐帧处理以获取BEV特征
            for i in range(len_queue):
                # 获取当前帧的元信息
                img_metas = [each[i] for each in img_metas_list]
                # 如果当前帧没有前一帧，重置prev_bev
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # 获取当前帧的图像特征
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # 通过BEVFormerHead获取BEV特征（只获取BEV，不进行检测）
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            
            # 恢复训练模式
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,              # 点云数据（BEVFormer中不使用）
                      img_metas=None,           # 图像元信息
                      gt_bboxes_3d=None,        # 真实3D边界框
                      gt_labels_3d=None,        # 真实3D标签
                      gt_labels=None,           # 真实2D标签（BEVFormer中不使用）
                      gt_bboxes=None,           # 真实2D边界框（BEVFormer中不使用）
                      img=None,                 # 输入图像，形状为(N, len_queue, num_cams, C, H, W)
                      proposals=None,           # 提议框（BEVFormer中不使用）
                      gt_bboxes_ignore=None,    # 要忽略的真实边界框
                      img_depth=None,           # 深度图（BEVFormer中不使用）
                      img_mask=None,            # 图像掩码（BEVFormer中不使用）
                      ):
        """训练时的前向传播函数
        
        Args:
            points (list[torch.Tensor], optional): 每个样本的点云数据
            img_metas (list[dict], optional): 每个样本的元信息
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional): 真实3D边界框
            gt_labels_3d (list[torch.Tensor], optional): 3D边界框的真实标签
            gt_labels (list[torch.Tensor], optional): 图像中2D边界框的真实标签
            gt_bboxes (list[torch.Tensor], optional): 图像中的真实2D边界框
            img (torch.Tensor, optional): 每个样本的图像，形状为(N, len_queue, num_cams, C, H, W)
            proposals ([list[torch.Tensor], optional): 用于训练Fast RCNN的预测提议
            gt_bboxes_ignore (list[torch.Tensor], optional): 要忽略的图像中的真实2D边界框
            
        Returns:
            dict: 不同分支的损失
        """
        
        # 获取时序队列长度
        len_queue = img.size(1)
        # 分离历史帧和当前帧
        prev_img = img[:, :-1, ...]  # 历史帧
        img = img[:, -1, ...]        # 当前帧

        # 深拷贝元信息以避免修改原始数据
        prev_img_metas = copy.deepcopy(img_metas)
        # 获取历史BEV特征
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        # 获取当前帧的元信息
        img_metas = [each[len_queue-1] for each in img_metas]
        # 如果当前帧没有前一帧，重置prev_bev
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        
        # 提取当前帧的图像特征
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # 初始化损失字典
        losses = dict()
        # 计算点云分支的损失（在BEVFormer中实际是图像分支）
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        # 更新总损失
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        """测试时的前向传播函数
        
        Args:
            img_metas (list[list[dict]]): 图像元信息
            img (list[torch.Tensor], optional): 输入图像
            
        Returns:
            list[dict]: 检测结果
        """
        # 验证输入格式
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # 检查是否是新场景的第一帧
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # 新场景的第一帧，重置前一帧BEV特征
            self.prev_frame_info['prev_bev'] = None
        # 更新场景标识符
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # 如果不使用时序信息，重置前一帧BEV特征
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # 获取两个时间戳之间自车位置和角度的变化量
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])      # 当前位置
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])    # 当前角度
        
        if self.prev_frame_info['prev_bev'] is not None:
            # 计算位置和角度的相对变化
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            # 如果没有前一帧，设置相对变化为0
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        # 执行简单测试
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        
        # 在推理过程中，保存每个时间戳的BEV特征和自车运动信息
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """点云分支的简单测试函数（在BEVFormer中处理图像特征）
        
        Args:
            x (list[torch.Tensor]): 输入特征
            img_metas (list[dict]): 图像元信息
            prev_bev (torch.Tensor, optional): 前一帧的BEV特征
            rescale (bool): 是否重新缩放边界框到原始图像尺寸
            
        Returns:
            tuple: (新的BEV特征, 边界框结果列表)
        """
        # 通过BEVFormerHead进行前向传播
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        # 获取边界框预测结果
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        
        # 将边界框结果转换为标准格式
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """无增强的简单测试函数
        
        Args:
            img_metas (list[dict]): 图像元信息
            img (torch.Tensor, optional): 输入图像
            prev_bev (torch.Tensor, optional): 前一帧的BEV特征
            rescale (bool): 是否重新缩放边界框
            
        Returns:
            tuple: (新的BEV特征, 边界框结果列表)
        """
        # 提取图像特征
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # 初始化结果列表
        bbox_list = [dict() for i in range(len(img_metas))]
        
        # 执行点云分支测试（实际处理图像特征）
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        
        # 组装最终结果
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return new_prev_bev, bbox_list
