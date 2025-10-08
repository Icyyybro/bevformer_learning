# 导入必要的库
import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS  # 重复导入，可以删除
import torch
import numpy as np  # 重复导入，可以删除
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    """自定义NuScenes数据集类
    
    这个数据集类继承自NuScenesDataset，主要添加了相机内参和外参信息到结果中。
    专门为BEVFormer模型设计，支持时序数据处理和BEV（鸟瞰图）特征提取。
    
    主要功能：
    1. 支持多帧时序数据的队列处理
    2. 处理相机内参外参变换矩阵
    3. 支持CAN总线数据的相对位置计算
    4. 提供自定义的评估方法
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        """初始化自定义NuScenes数据集
        
        Args:
            queue_length (int): 时序队列长度，用于多帧数据处理。默认为4
            bev_size (tuple): BEV特征图尺寸，格式为(height, width)。默认为(200, 200)
            overlap_test (bool): 是否进行重叠测试。默认为False
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length  # 时序队列长度
        self.overlap_test = overlap_test  # 重叠测试标志
        self.bev_size = bev_size  # BEV特征图尺寸
        
    def prepare_train_data(self, index):
        """准备训练数据
        
        这个方法为BEVFormer准备时序训练数据，通过构建一个包含多帧历史数据的队列，
        支持时序特征学习和BEV特征的时间一致性。
        
        Args:
            index (int): 当前帧的索引
            
        Returns:
            dict: 处理后的训练数据字典，包含多帧图像和元数据
            None: 如果数据无效则返回None
        """
        queue = []  # 存储多帧数据的队列
        
        # 构建时序索引列表：从(index-queue_length)到index
        index_list = list(range(index-self.queue_length, index))
        
        # 随机打乱历史帧顺序（除了当前帧），增加数据多样性
        random.shuffle(index_list)
        
        # 丢掉随机排序后的第一帧，并重新排序历史帧，保持时间顺序
        index_list = sorted(index_list[1:])
        
        # 添加当前帧到队列末尾
        index_list.append(index)
        
        # 处理队列中的每一帧
        for i in index_list:
            # 确保索引不为负数
            i = max(0, i)
            
            # 获取第i帧的数据信息
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
                
            # 应用预处理管道
            self.pre_pipeline(input_dict)
            
            # 应用数据处理管道
            example = self.pipeline(input_dict)
            
            # 检查是否需要过滤空的ground truth
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
                
            queue.append(example)
            
        # 将多帧数据合并为一个数据样本
        return self.union2one(queue)


    def union2one(self, queue):
        """将多帧数据合并为单个训练样本
        
        这个方法是BEVFormer的核心数据处理函数，负责：
        1. 合并多帧图像数据
        2. 计算相对位置和角度变化
        3. 设置BEV特征是否存在的标志
        
        Args:
            queue (list): 包含多帧数据的队列
            
        Returns:
            dict: 合并后的数据字典
        """
        # 提取所有帧的图像数据
        imgs_list = [each['img'].data for each in queue]
        
        # 存储每帧的元数据
        metas_map = {}
        
        # 用于跟踪场景变化的变量
        prev_scene_token = None  # 前一帧的场景token
        prev_pos = None          # 前一帧的位置
        prev_angle = None        # 前一帧的角度
        
        # 处理每一帧的元数据
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            
            # 检查是否为新场景的第一帧（NuScenes数据集中，每个 scene_token 代表一个连续的驾驶场景（通常持续20秒））
            if metas_map[i]['scene_token'] != prev_scene_token:
                # 新场景：设置BEV不存在，重置相对位置
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                
                # 保存当前帧的绝对位置和角度作为参考
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                
                # 将当前帧的相对位置设为0（作为参考帧）
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                # 同一场景：计算相对于前一帧的位置变化
                # 这是BEVFormer时序建模的关键步骤，通过计算帧间相对变化来实现BEV特征的时间对齐
                metas_map[i]['prev_bev_exists'] = True
                
                # 步骤1：保存当前帧的绝对位置和角度（在修改之前备份）
                # 这些是当前帧在全局坐标系下的真实位置和朝向
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])      # 当前帧绝对位置 [x, y, z]
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])    # 当前帧绝对角度（度数）
                
                # 步骤2：计算相对位置和角度变化
                # 将当前帧的绝对位置转换为相对于前一帧的相对位置
                # 这样做的目的是让模型学习帧间的运动变化，而不是绝对位置
                metas_map[i]['can_bus'][:3] -= prev_pos    # 相对位移 = 当前位置 - 前一帧位置
                metas_map[i]['can_bus'][-1] -= prev_angle  # 相对角度变化 = 当前角度 - 前一帧角度
                
                # 步骤3：更新参考位置和角度，为下一帧的计算做准备
                # 将当前帧的绝对位置设为新的参考系，供下一帧使用
                prev_pos = copy.deepcopy(tmp_pos)      # 更新参考位置为当前帧的绝对位置
                prev_angle = copy.deepcopy(tmp_angle)  # 更新参考角度为当前帧的绝对角度
                
                # 经过这个处理后：
                # - metas_map[i]['can_bus'][:3] 存储的是相对于前一帧的位移向量
                # - metas_map[i]['can_bus'][-1] 存储的是相对于前一帧的角度变化
                # - prev_pos, prev_angle 被更新为当前帧的绝对值，供下一帧计算使用
        
        # 将多帧图像堆叠为一个张量
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        
        # 将元数据字典封装为DataContainer
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        
        # 返回最后一帧作为主要数据，但包含了所有帧的信息
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """根据给定索引获取数据信息
        
        这个方法从数据集中提取指定索引的样本信息，包括：
        - 点云数据路径
        - 图像数据路径
        - 相机内参外参
        - 坐标变换矩阵
        - CAN总线数据
        - 标注信息

        Args:
            index (int): 要获取的样本数据索引

        Returns:
            dict: 将传递给数据预处理管道的数据信息字典，包含以下键值：
                - sample_idx (str): 样本索引
                - pts_filename (str): 点云文件名
                - sweeps (list[dict]): 扫描信息
                - timestamp (float): 样本时间戳
                - img_filename (str, optional): 图像文件名
                - lidar2img (list[np.ndarray], optional): 从激光雷达到不同相机的变换矩阵
                - ann_info (dict): 标注信息
        """
        # 获取指定索引的数据信息
        info = self.data_infos[index]
        
        # 构建基础输入字典（遵循SECOND.Pytorch的标准协议）
        input_dict = dict(
            sample_idx=info['token'],                           # 样本唯一标识符
            pts_filename=info['lidar_path'],                    # 点云文件路径
            sweeps=info['sweeps'],                              # 激光雷达扫描数据
            ego2global_translation=info['ego2global_translation'], # 车体到全局坐标的平移
            ego2global_rotation=info['ego2global_rotation'],       # 车体到全局坐标的旋转
            prev_idx=info['prev'],                              # 前一帧索引
            next_idx=info['next'],                              # 后一帧索引
            scene_token=info['scene_token'],                    # 场景标识符
            can_bus=info['can_bus'],                           # CAN总线数据
            frame_idx=info['frame_idx'],                       # 帧索引
            timestamp=info['timestamp'] / 1e6,                 # 时间戳（转换为秒）
        )

        # 如果使用相机数据，处理相机相关信息
        if self.modality['use_camera']:
            image_paths = []      # 图像路径列表
            lidar2img_rts = []    # 激光雷达到图像的变换矩阵列表
            lidar2cam_rts = []    # 激光雷达到相机的变换矩阵列表
            cam_intrinsics = []   # 相机内参矩阵列表
            
            # 遍历所有相机
            for cam_type, cam_info in info['cams'].items():
                # 添加图像路径
                image_paths.append(cam_info['data_path'])
                
                # 计算激光雷达到图像的变换矩阵
                # 1. 获取激光雷达到相机的旋转矩阵（取逆）
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                
                # 2. 计算激光雷达到相机的平移向量
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                
                # 3. 构建4x4齐次变换矩阵
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T  # 旋转部分
                lidar2cam_rt[3, :3] = -lidar2cam_t    # 平移部分
                
                # 4. 获取相机内参矩阵
                intrinsic = cam_info['cam_intrinsic']
                
                # 5. 将3x3内参矩阵扩展为4x4矩阵
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                
                # 6. 计算最终的激光雷达到图像变换矩阵
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                # 保存相机内参和外参
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            # 更新输入字典，添加相机相关信息
            input_dict.update(
                dict(
                    img_filename=image_paths,      # 图像文件路径列表
                    lidar2img=lidar2img_rts,      # 激光雷达到图像变换矩阵列表
                    cam_intrinsic=cam_intrinsics,  # 相机内参矩阵列表
                    lidar2cam=lidar2cam_rts,      # 激光雷达到相机变换矩阵列表
                ))

        # 如果不是测试模式，添加标注信息
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        # 处理CAN总线数据，计算车辆的位置和朝向信息
        rotation = Quaternion(input_dict['ego2global_rotation'])  # 四元数旋转
        translation = input_dict['ego2global_translation']        # 平移向量
        can_bus = input_dict['can_bus']                          # CAN总线数据
        
        # 设置CAN总线数据的位置信息（前3个元素）
        can_bus[:3] = translation
        
        # 设置CAN总线数据的旋转信息（第4-7个元素）
        can_bus[3:7] = rotation
        
        # 计算车辆朝向角度
        patch_angle = quaternion_yaw(rotation) / np.pi * 180  # 转换为度数
        if patch_angle < 0:
            patch_angle += 360  # 确保角度在0-360度范围内
            
        # 设置CAN总线数据的角度信息
        can_bus[-2] = patch_angle / 180 * np  # 弧度制角度
        can_bus[-1] = patch_angle                # 度数制角度

        return input_dict

    def __getitem__(self, idx):
        """根据给定索引获取数据项
        
        这是数据集的核心访问方法，支持训练和测试两种模式。
        在训练模式下，如果数据无效会自动重新采样。
        
        Args:
            idx (int): 数据索引
            
        Returns:
            dict: 对应索引的数据字典
        """
        # 测试模式：直接返回测试数据
        if self.test_mode:
            return self.prepare_test_data(idx)
            
        # 训练模式：循环直到获取有效数据
        while True:
            # 准备训练数据
            data = self.prepare_train_data(idx)
            
            # 如果数据无效，随机选择另一个索引
            if data is None:
                idx = self._rand_another(idx)
                continue
                
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """使用nuScenes协议对单个模型进行评估
        
        这个方法实现了nuScenes数据集的标准评估流程，包括：
        1. 加载nuScenes数据集
        2. 初始化自定义评估器
        3. 运行评估并生成指标
        4. 解析和格式化评估结果

        Args:
            result_path (str): 结果文件路径
            logger (logging.Logger | str | None): 用于打印评估相关信息的日志器。默认为None
            metric (str): 用于评估的指标名称。默认为'bbox'
            result_name (str): 指标前缀中的结果名称。默认为'pts_bbox'

        Returns:
            dict: 评估详情字典，包含各类别和各距离阈值的AP值、TP错误等指标
        """
        # 导入并初始化nuScenes数据集
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        # 获取输出目录
        output_dir = osp.join(*osp.split(result_path)[:-1])

        # 数据集版本到评估集的映射
        eval_set_map = {
            'v1.0-mini': 'mini_val',      # 迷你版本对应迷你验证集
            'v1.0-trainval': 'val',       # 训练验证版本对应验证集
        }
        
        # 初始化自定义nuScenes评估器
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,                                    # nuScenes数据集对象
            config=self.eval_detection_configs,          # 检测评估配置
            result_path=result_path,                      # 结果文件路径
            eval_set=eval_set_map[self.version],         # 评估集名称
            output_dir=output_dir,                        # 输出目录
            verbose=True,                                 # 详细输出
            overlap_test=self.overlap_test,               # 重叠测试标志
            data_infos=self.data_infos                   # 数据信息
        )
        
        # 运行评估（不绘制示例，不渲染曲线）
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        
        # 记录评估指标
        # 加载评估结果
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()  # 存储详细评估结果
        metric_prefix = f'{result_name}_NuScenes'  # 指标前缀
        
        # 遍历所有类别，收集各类别的AP和TP错误指标
        for name in self.CLASSES:
            # 收集各距离阈值下的AP值
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
                
            # 收集各类别的TP错误指标
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                
        # 收集总体TP错误指标
        for k, v in metrics['tp_errors'].items():
            val = float('{:.4f}'.format(v))
            detail['{}/{}'.format(metric_prefix,
                                  self.ErrNameMapping[k])] = val
                                  
        # 添加NDS（nuScenes Detection Score）和mAP指标
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']  # nuScenes检测分数
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']   # 平均精度
        
        return detail
