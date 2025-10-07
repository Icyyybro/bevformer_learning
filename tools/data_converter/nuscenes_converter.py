# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """创建 nuScenes 数据集的信息文件。

    根据原始数据，生成相关的 pkl 格式信息文件。

    Args:
        root_path (str): 数据根目录路径。
        out_path (str): 输出路径，用于保存生成的信息文件。
        can_bus_root_path (str): CAN 总线数据的根目录路径。
        info_prefix (str): 要生成的信息文件的前缀。
        version (str): 数据版本。
            默认值: 'v1.0-trainval'
        max_sweeps (int): 最大扫描帧数。
            默认值: 10
    """
    # 导入 nuScenes 相关的库
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    
    # 打印版本和根路径信息
    print(version, root_path)
    
    # 初始化 nuScenes 数据集对象，加载数据集元数据
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    
    # 初始化 CAN 总线数据对象，用于获取车辆状态信息
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    
    # 导入数据集划分信息
    from nuscenes.utils import splits
    
    # 定义支持的数据集版本
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    
    # 根据版本选择对应的场景划分
    if version == 'v1.0-trainval':
        # 完整训练验证集：包含训练和验证场景
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        # 测试集：只有测试场景，无验证集
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        # 迷你版本：用于快速测试和开发
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # 过滤现有场景，确保场景文件实际存在
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    
    # 过滤训练场景：只保留实际存在的场景
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    
    # 过滤验证场景：只保留实际存在的场景
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    
    # 将场景名称转换为场景令牌（token），用于后续数据处理
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    # 判断是否为测试模式
    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    # 填充训练和验证信息，这是核心的数据处理步骤
    # 该函数会提取每个样本的详细信息，包括：
    # - 传感器数据路径（相机、激光雷达等）
    # - 标注信息（3D边界框、类别等）
    # - 相机内外参数
    # - 车辆状态信息（来自CAN总线）
    # - 时序信息（多帧扫描数据）
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    # 创建元数据字典，包含版本信息
    metadata = dict(version=version)
    
    if test:
        # 测试模式：只保存测试数据
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        # 训练模式：分别保存训练和验证数据
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        
        # 保存训练数据信息文件
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        
        # 保存验证数据信息文件
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """从输入的 nuScenes 类中获取可用场景。

    根据原始数据，获取可用场景的信息，用于进一步的信息生成。
    该函数会验证场景对应的数据文件是否真实存在于磁盘上。

    Args:
        nusc (class): nuScenes 数据集类对象。

    Returns:
        available_scenes (list[dict]): 可用场景的基本信息列表。
            每个场景包含场景令牌、名称等元数据。
    """
    # 初始化可用场景列表
    available_scenes = []
    
    # 打印数据集中的总场景数量
    print('total scene num: {}'.format(len(nusc.scene)))
    
    # 遍历数据集中的每个场景
    for scene in nusc.scene:
        # 获取场景的唯一标识符
        scene_token = scene['token']
        
        # 通过场景令牌获取场景记录
        scene_rec = nusc.get('scene', scene_token)
        
        # 获取场景中第一个样本的记录
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        
        # 获取第一个样本中激光雷达顶部传感器的数据记录
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        
        # 初始化标志变量
        has_more_frames = True  # 是否还有更多帧（当前未使用，但保留逻辑结构）
        scene_not_exist = False  # 场景数据是否不存在
        
        # 检查场景数据文件是否存在
        # 注意：这里的 while 循环实际上只执行一次，用于检查第一帧数据
        while has_more_frames:
            # 获取激光雷达数据的文件路径和边界框信息
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            
            # 处理路径格式：如果是绝对路径且包含当前工作目录
            if os.getcwd() in lidar_path:
                # 来自 lyft 数据集的路径是绝对路径，需要转换为相对路径
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # 转换为相对路径
            
            # 检查激光雷达数据文件是否真实存在
            if not mmcv.is_filepath(lidar_path):
                # 如果文件不存在，标记该场景为不可用
                scene_not_exist = True
                break
            else:
                # 如果文件存在，跳出循环（只检查第一帧就足够了）
                break
        
        # 如果场景数据不存在，跳过该场景
        if scene_not_exist:
            continue
        
        # 如果场景数据存在，将其添加到可用场景列表中
        available_scenes.append(scene)
    
    # 打印实际存在的场景数量
    print('exist scene num: {}'.format(len(available_scenes)))
    
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    """从CAN总线数据中获取车辆状态信息。
    
    CAN总线（Controller Area Network）记录了车辆的实时状态信息，
    包括位置、方向、速度、加速度、转向角等重要的车辆动态参数。
    这些信息对于自动驾驶模型理解车辆运动状态非常重要。
    
    Args:
        nusc: nuScenes数据集对象
        nusc_can_bus: CAN总线数据对象
        sample: 当前样本数据
        
    Returns:
        np.array: 长度为18的CAN总线信息数组，包含车辆的完整状态信息
    """
    # 获取当前样本所属场景的名称
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    
    # 获取当前样本的时间戳（微秒）
    sample_timestamp = sample['timestamp']
    
    try:
        # 尝试获取该场景的所有姿态（pose）消息
        # pose消息包含车辆的位置、方向和其他状态信息
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        # 如果获取失败（某些服务器场景没有CAN总线信息），返回零向量
        return np.zeros(18)  # 服务器场景没有CAN总线信息
    
    # 初始化CAN总线信息列表
    can_bus = []
    
    # 在每个场景中，CAN总线的第一个时间戳可能大于第一个样本的时间戳
    # 因此需要找到时间戳最接近且不超过当前样本时间戳的姿态数据
    last_pose = pose_list[0]  # 初始化为第一个姿态
    
    # 遍历所有姿态消息，找到时间上最接近的姿态
    for i, pose in enumerate(pose_list):
        # 如果当前姿态的时间戳超过了样本时间戳，停止搜索
        if pose['utime'] > sample_timestamp:
            break
        # 更新最近的姿态
        last_pose = pose
    
    # 从姿态数据中提取信息并构建CAN总线向量
    
    # 移除时间戳（不需要包含在最终向量中）
    _ = last_pose.pop('utime')  # 时间戳信息，已无用
    
    # 提取并移除位置信息 (x, y, z) - 3个元素
    pos = last_pose.pop('pos')
    
    # 提取并移除方向信息（四元数或欧拉角） - 通常4个元素
    rotation = last_pose.pop('orientation')
    
    # 将位置信息添加到CAN总线向量中
    can_bus.extend(pos)
    
    # 将方向信息添加到CAN总线向量中
    can_bus.extend(rotation)
    
    # 添加剩余的所有车辆状态信息
    # 这些可能包括：速度、加速度、转向角、刹车状态等
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 总共16个元素
    
    # 添加两个额外的零元素，使总长度达到18
    # 这可能是为了与模型期望的输入维度保持一致
    can_bus.extend([0., 0.])
    
    # 转换为numpy数组并返回
    # 最终的CAN总线向量包含：
    # - 位置信息 (3个元素)
    # - 方向信息 (4个元素) 
    # - 其他车辆状态 (16个元素中的11个)
    # - 填充零 (2个元素)
    # 总计：18个元素
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """从原始数据生成训练/验证信息。

    Args:
        nusc (:obj:`NuScenes`): nuScenes 数据集类对象。
        nusc_can_bus: CAN 总线数据对象，用于获取车辆状态信息。
        train_scenes (list[str]): 训练场景的基本信息。
        val_scenes (list[str]): 验证场景的基本信息。
        test (bool): 是否使用测试模式。在测试模式下，无法访问标注信息。默认值: False。
        max_sweeps (int): 最大扫描帧数，用于时序信息。默认值: 10。

    Returns:
        tuple[list[dict]]: 将保存到信息文件中的训练集和验证集信息。
    """
    # 初始化训练和验证信息列表
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0  # 帧索引，用于跟踪时序信息
    
    # 遍历数据集中的所有样本，显示进度条
    for sample in mmcv.track_iter_progress(nusc.sample):
        # 获取激光雷达顶部传感器的令牌
        lidar_token = sample['data']['LIDAR_TOP']
        
        # 获取激光雷达传感器数据记录
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        
        # 获取校准传感器记录（包含传感器到车辆的变换）
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        
        # 获取车辆姿态记录（包含车辆到全局坐标系的变换）
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        
        # 获取激光雷达数据路径、3D边界框和其他信息
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        # 检查激光雷达文件是否存在
        mmcv.check_file_exist(lidar_path)
        
        # 获取 CAN 总线信息（车辆状态：速度、转向角等）
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        
        # 构建样本信息字典，包含所有必要的元数据
        info = {
            'lidar_path': lidar_path,  # 激光雷达数据文件路径
            'token': sample['token'],  # 样本唯一标识符
            'prev': sample['prev'],    # 前一帧样本的令牌
            'next': sample['next'],    # 后一帧样本的令牌
            'can_bus': can_bus,        # CAN 总线数据（车辆状态信息）
            'frame_idx': frame_idx,    # 时序相关信息：当前帧在序列中的索引
            'sweeps': [],              # 扫描帧列表（用于多帧融合）
            'cams': dict(),            # 相机信息字典
            'scene_token': sample['scene_token'],  # 时序相关信息：场景令牌
            'lidar2ego_translation': cs_record['translation'],  # 激光雷达到车辆坐标系的平移
            'lidar2ego_rotation': cs_record['rotation'],        # 激光雷达到车辆坐标系的旋转
            'ego2global_translation': pose_record['translation'],  # 车辆到全局坐标系的平移
            'ego2global_rotation': pose_record['rotation'],        # 车辆到全局坐标系的旋转
            'timestamp': sample['timestamp'],  # 时间戳
        }

        # 更新帧索引：如果是序列的最后一帧，重置为0；否则递增
        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        # 提取变换矩阵，用于坐标系转换
        l2e_r = info['lidar2ego_rotation']      # 激光雷达到车辆的旋转
        l2e_t = info['lidar2ego_translation']   # 激光雷达到车辆的平移
        e2g_r = info['ego2global_rotation']     # 车辆到全局的旋转
        e2g_t = info['ego2global_translation']  # 车辆到全局的平移
        
        # 将四元数转换为旋转矩阵，便于后续计算
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # 获取每帧6个相机的信息
        camera_types = [
            'CAM_FRONT',        # 前置相机
            'CAM_FRONT_RIGHT',  # 右前相机
            'CAM_FRONT_LEFT',   # 左前相机
            'CAM_BACK',         # 后置相机
            'CAM_BACK_LEFT',    # 左后相机
            'CAM_BACK_RIGHT',   # 右后相机
        ]
        
        # 遍历每个相机，提取相机信息
        for cam in camera_types:
            cam_token = sample['data'][cam]  # 获取相机数据令牌
            
            # 获取相机图像路径和内参矩阵
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            
            # 计算相机到激光雷达顶部的变换关系
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            
            # 添加相机内参信息
            cam_info.update(cam_intrinsic=cam_intrinsic)
            
            # 将相机信息添加到字典中
            info['cams'].update({cam: cam_info})

        # 获取单个关键帧的扫描帧信息（用于时序融合）
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        
        # 向前追溯最多 max_sweeps 帧的历史扫描数据
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':  # 如果存在前一帧
                # 获取前一帧激光雷达到当前帧激光雷达的变换关系
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                
                # 移动到更早的一帧
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break  # 没有更早的帧了
        
        info['sweeps'] = sweeps
        
        # 获取标注信息（仅在非测试模式下）
        if not test:
            # 获取所有标注对象
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            
            # 提取3D边界框的位置信息 (x, y, z)
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            
            # 提取3D边界框的尺寸信息 (width, length, height)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            
            # 提取3D边界框的旋转角度（偏航角）
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            
            # 提取物体的速度信息（仅x, y方向）
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            
            # 创建有效标志：只有被激光雷达或雷达检测到的物体才有效
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            
            # 将速度从全局坐标系转换到激光雷达坐标系
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])  # 添加z方向速度（为0）
                
                # 应用逆变换：全局 -> 车辆 -> 激光雷达
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]  # 只保留x, y方向的速度

            # 提取物体类别名称
            names = [b.name for b in boxes]
            
            # 将类别名称映射到标准名称（如果存在映射关系）
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            
            # 将旋转角度转换为 SECOND 格式（MMDetection3D 使用的格式）
            # 注意：这里进行了角度转换和坐标系调整
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            
            # 确保边界框数量与标注数量一致
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            
            # 添加标注信息到样本信息中
            info['gt_boxes'] = gt_boxes        # 3D边界框 [x, y, z, w, l, h, rot]
            info['gt_names'] = names           # 物体类别名称
            info['gt_velocity'] = velocity.reshape(-1, 2)  # 物体速度 [vx, vy]
            info['num_lidar_pts'] = np.array(  # 每个物体的激光雷达点数
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(  # 每个物体的雷达点数
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag    # 有效标志

        # 根据场景令牌将样本分配到训练集或验证集
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """获取从通用传感器到顶部激光雷达的RT变换矩阵信息。

    该函数计算任意传感器（相机、激光雷达等）到参考激光雷达（Top LiDAR）的
    空间变换关系，这对于多传感器数据融合至关重要。

    坐标系变换链：
    传感器坐标系 -> 车辆坐标系 -> 全局坐标系 -> 参考车辆坐标系 -> 参考激光雷达坐标系

    Args:
        nusc (class): nuScenes 数据集类对象。
        sensor_token (str): 对应特定传感器类型的样本数据令牌。
        l2e_t (np.ndarray): 参考激光雷达到车辆坐标系的平移向量，形状 (1, 3)。
        l2e_r_mat (np.ndarray): 参考激光雷达到车辆坐标系的旋转矩阵，形状 (3, 3)。
        e2g_t (np.ndarray): 参考车辆到全局坐标系的平移向量，形状 (1, 3)。
        e2g_r_mat (np.ndarray): 参考车辆到全局坐标系的旋转矩阵，形状 (3, 3)。
        sensor_type (str): 要校准的传感器类型。默认值: 'lidar'。

    Returns:
        sweep (dict): 变换后的扫描信息，包含传感器到激光雷达的变换矩阵。
    """
    # 获取传感器数据记录
    sd_rec = nusc.get('sample_data', sensor_token)
    
    # 获取校准传感器记录（包含传感器到车辆的变换）
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    
    # 获取车辆姿态记录（包含车辆到全局坐标系的变换）
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # 获取传感器数据文件路径
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    
    # 处理路径格式：如果是绝对路径，转换为相对路径
    if os.getcwd() in data_path:  # 来自 lyft 数据集的路径是绝对路径
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # 转换为相对路径
    
    # 构建扫描信息字典，包含基本的传感器信息和变换参数
    sweep = {
        'data_path': data_path,                              # 数据文件路径
        'type': sensor_type,                                 # 传感器类型
        'sample_data_token': sd_rec['token'],                # 样本数据令牌
        'sensor2ego_translation': cs_record['translation'],   # 传感器到车辆的平移
        'sensor2ego_rotation': cs_record['rotation'],        # 传感器到车辆的旋转
        'ego2global_translation': pose_record['translation'], # 车辆到全局的平移
        'ego2global_rotation': pose_record['rotation'],      # 车辆到全局的旋转
        'timestamp': sd_rec['timestamp']                     # 时间戳
    }

    # 提取当前传感器的变换参数（简化变量名）
    l2e_r_s = sweep['sensor2ego_rotation']      # 当前传感器到车辆的旋转
    l2e_t_s = sweep['sensor2ego_translation']   # 当前传感器到车辆的平移
    e2g_r_s = sweep['ego2global_rotation']      # 当前车辆到全局的旋转
    e2g_t_s = sweep['ego2global_translation']   # 当前车辆到全局的平移

    # 计算从传感器到顶部激光雷达的RT变换矩阵
    # 变换链：传感器 -> 车辆 -> 全局 -> 参考车辆 -> 参考激光雷达
    
    # 将四元数转换为旋转矩阵
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix  # 当前传感器到车辆的旋转矩阵
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix  # 当前车辆到全局的旋转矩阵
    
    # 计算旋转矩阵 R
    # 变换顺序：传感器 -> 车辆 -> 全局 -> 参考车辆 -> 参考激光雷达
    # R = (传感器->车辆)^T @ (车辆->全局)^T @ (全局->参考车辆) @ (参考车辆->参考激光雷达)
    # 其中 (全局->参考车辆) = (参考车辆->全局)^(-1)
    # (参考车辆->参考激光雷达) = (参考激光雷达->参考车辆)^(-1)
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    
    # 计算平移向量 T
    # 首先将传感器位置变换到全局坐标系
    # 然后变换到参考激光雷达坐标系
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    
    # 减去参考点在参考激光雷达坐标系中的位置
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    
    # 保存最终的变换矩阵
    # 注意：这里使用 R.T 是因为点变换的公式是 points @ R.T + T
    sweep['sensor2lidar_rotation'] = R.T        # 传感器到激光雷达的旋转矩阵
    sweep['sensor2lidar_translation'] = T       # 传感器到激光雷达的平移向量
    
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """从信息文件和原始数据导出2D标注。

    该函数将3D边界框投影到各个相机的2D图像平面上，生成COCO格式的2D标注文件。
    这对于2D目标检测、单目3D检测等任务非常有用。

    Args:
        root_path (str): 原始数据的根路径。
        info_path (str): 信息文件的路径（pkl格式）。
        version (str): 数据集版本。
        mono3d (bool): 是否导出单目3D标注信息。默认值: True。
            如果为True，会包含3D边界框的投影信息；
            如果为False，只包含2D边界框信息。
    """
    # 定义6个相机类型，nuScenes数据集的标准相机配置
    camera_types = [
        'CAM_FRONT',        # 前置相机
        'CAM_FRONT_RIGHT',  # 右前相机
        'CAM_FRONT_LEFT',   # 左前相机
        'CAM_BACK',         # 后置相机
        'CAM_BACK_LEFT',    # 左后相机
        'CAM_BACK_RIGHT',   # 右后相机
    ]
    
    # 加载预处理好的信息文件
    nusc_infos = mmcv.load(info_path)['infos']
    
    # 初始化nuScenes数据集对象
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    
    # 创建类别ID映射，将类别名称映射到数字ID
    # 这是COCO格式要求的类别信息
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    
    # 初始化COCO标注ID计数器，每个标注都需要唯一ID
    coco_ann_id = 0
    
    # 初始化COCO格式的数据字典
    # COCO格式包含三个主要部分：annotations（标注）、images（图像）、categories（类别）
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    
    # 遍历所有样本信息，显示进度条
    for info in mmcv.track_iter_progress(nusc_infos):
        # 遍历每个相机
        for cam in camera_types:
            # 获取当前相机的信息
            cam_info = info['cams'][cam]
            
            # 获取当前相机的2D边界框信息
            # visibilities参数控制可见性级别：
            # '': 未知, '1': 0-40%, '2': 40-60%, '3': 60-80%, '4': 80-100%
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],  # 包含所有可见性级别
                mono3d=mono3d)
            
            # 读取图像获取尺寸信息
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            
            # 添加图像信息到COCO字典
            coco_2d_dict['images'].append(
                dict(
                    # 提取相对路径作为文件名
                    file_name=cam_info['data_path'].split('data/nuscenes/')[-1],
                    id=cam_info['sample_data_token'],  # 使用样本数据令牌作为图像ID
                    token=info['token'],               # 样本令牌
                    
                    # 相机外参：相机到车辆坐标系的变换
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    
                    # 车辆姿态：车辆到全局坐标系的变换
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    
                    # 相机内参矩阵
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    
                    # 图像尺寸
                    width=width,
                    height=height))
            
            # 处理当前相机的所有2D边界框标注
            for coco_info in coco_infos:
                if coco_info is None:
                    continue  # 跳过无效的标注
                
                # 为COCO格式添加分割信息（这里为空，因为我们只做检测）
                coco_info['segmentation'] = []
                
                # 分配唯一的标注ID
                coco_info['id'] = coco_ann_id
                
                # 添加标注到COCO字典
                coco_2d_dict['annotations'].append(coco_info)
                
                # 递增标注ID计数器
                coco_ann_id += 1
    
    # 根据是否包含单目3D信息确定输出文件名
    if mono3d:
        # 包含3D信息的文件名
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        # 只包含2D信息的文件名
        json_prefix = f'{info_path[:-4]}'
    
    # 保存COCO格式的2D标注文件
    # 输出文件将包含：
    # - images: 图像信息和相机参数
    # - annotations: 2D边界框标注（可能包含3D投影信息）
    # - categories: 物体类别定义
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """获取给定 `sample_data_token` 的2D标注记录。

    该函数将3D边界框投影到2D图像平面，生成2D检测标注。
    这是连接3D世界和2D图像的关键桥梁，支持单目3D检测等任务。

    Args:
        sample_data_token (str): 属于相机关键帧的样本数据令牌。
        visibilities (list[str]): 可见性过滤器，控制包含哪些可见性级别的物体。
        mono3d (bool): 是否获取包含单目3D标注的边界框。

    Return:
        list[dict]: 属于输入 `sample_data_token` 的2D标注记录列表。
    """

    # 获取样本数据记录和对应的样本记录
    sd_rec = nusc.get('sample_data', sample_data_token)

    # 确保这是相机数据，该函数只适用于相机传感器
    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    
    # 确保这是关键帧，2D重投影只对关键帧可用
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    # 获取对应的样本记录
    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # 获取校准传感器和车辆姿态记录，用于获取变换矩阵
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # 获取相机内参矩阵，用于3D到2D的投影
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # 获取所有指定可见性的标注
    # 首先获取所有标注记录
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    
    # 根据可见性过滤标注
    # 只保留可见性符合要求的物体
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    # 初始化重投影记录列表
    repro_recs = []

    # 遍历每个标注记录
    for ann_rec in ann_recs:
        # 为样本标注添加令牌信息
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # 获取全局坐标系下的3D边界框
        box = nusc.get_box(ann_rec['token'])

        # 坐标系变换：全局坐标系 -> 车辆坐标系
        # 先平移再旋转（逆变换）
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # 坐标系变换：车辆坐标系 -> 相机坐标系
        # 先平移再旋转（逆变换）
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # 过滤掉不在相机前方的角点
        # 获取3D边界框的8个角点
        corners_3d = box.corners()
        
        # 找到Z坐标大于0的角点（相机前方）
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # 将3D边界框投影到2D图像平面
        # 使用相机内参进行透视投影
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # 只保留落在图像范围内的角点
        # 并计算包围所有有效角点的最小外接矩形
        final_coords = post_process_coords(corner_coords)

        # 如果重投影角点的凸包不与图像画布相交，跳过该物体
        if final_coords is None:
            continue
        else:
            # 获取2D边界框的坐标
            min_x, min_y, max_x, max_y = final_coords

        # 生成要包含在.json文件中的字典记录
        # 创建基本的2D标注记录
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # 如果 mono3d=True，添加相机坐标系下的3D标注
        if mono3d and (repro_rec is not None):
            # 获取3D边界框中心位置（相机坐标系）
            loc = box.center.tolist()

            # 获取3D边界框尺寸并转换格式
            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # 将 wlh 转换为 lhw 格式
            dim = dim.tolist()

            # 获取旋转角度并转换到相机坐标系
            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # 转换旋转角度到相机坐标系

            # 计算相机坐标系下的速度
            # 首先获取全局坐标系下的2D速度
            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            
            # 获取变换矩阵
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            
            # 将速度从全局坐标系转换到相机坐标系
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()  # 只取x, y方向的速度

            # 添加3D边界框信息（相机坐标系）
            # 格式：[x, y, z, l, h, w, rot]
            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            # 计算3D中心点在2D图像上的投影
            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            
            # 标准化的2D中心点 + 深度信息
            # 如果深度 < 0 的样本将被移除（在相机后方）
            if repro_rec['center2d'][2] <= 0:
                continue

            # 获取物体属性信息
            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                # 如果没有属性，设为 'None'
                attr_name = 'None'
            else:
                # 获取第一个属性的名称
                attr_name = nusc.get('attribute', ann_token[0])['name']
            
            # 获取属性ID
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        # 将处理好的记录添加到结果列表
        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """
    计算重投影边界框角点的凸包与图像画布的交集，如果没有交集则返回None，如果有交集则返回交集的边界框坐标（BBox）
    
    该函数是3D到2D投影后处理的关键步骤，用于：
    1. 处理3D边界框投影到2D图像平面后可能出现的不规则形状
    2. 确保最终的2D边界框在图像范围内
    3. 处理部分遮挡或超出图像边界的情况
    
    Args:
        corner_coords (list[int]): 重投影后的边界框角点坐标列表
                                  通常是3D边界框8个角点投影到2D平面的结果
        imsize (tuple[int]): 图像画布的尺寸 (宽度, 高度)
                           默认为 (1600, 900)，对应nuScenes相机图像尺寸

    Return:
        tuple [float]: 凸包与图像画布交集的边界框坐标 (min_x, min_y, max_x, max_y)
                      如果没有交集则返回None
    """
    # 将角点坐标转换为多点几何对象，并计算其凸包
    # 凸包是包含所有点的最小凸多边形，用于处理投影后可能出现的不规则形状
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    
    # 创建图像画布的矩形几何对象
    # box(minx, miny, maxx, maxy) 创建一个矩形
    img_canvas = box(0, 0, imsize[0], imsize[1])

    # 检查边界框凸包是否与图像画布有交集
    if polygon_from_2d_box.intersects(img_canvas):
        # 计算凸包与图像画布的交集区域
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        
        # 提取交集区域的边界坐标点
        # exterior.coords 获取多边形外边界的坐标点
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        # 计算交集区域的边界框坐标
        # 找到所有交集点中的最小和最大x、y坐标
        min_x = min(intersection_coords[:, 0])  # 左边界
        min_y = min(intersection_coords[:, 1])  # 上边界  
        max_x = max(intersection_coords[:, 0])  # 右边界
        max_y = max(intersection_coords[:, 1])  # 下边界

        # 返回标准的边界框格式：(左, 上, 右, 下)
        return min_x, min_y, max_x, max_y
    else:
        # 如果没有交集，说明3D物体完全在图像范围外，返回None
        # 这种情况下该物体在当前相机视角下不可见
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """
    根据2D边界框坐标和其他信息生成一个2D标注记录
    
    该函数是数据格式转换的关键环节，将nuScenes原始标注转换为COCO格式。
    主要功能包括：
    1. 提取相关的标注信息
    2. 计算2D边界框的面积
    3. 进行类别名称映射和ID转换
    4. 生成符合COCO格式要求的标注记录

    Args:
        ann_rec (dict): 原始的3D标注记录，包含nuScenes格式的完整标注信息
        x1 (float): x坐标的最小值（左边界）
        y1 (float): y坐标的最小值（上边界）
        x2 (float): x坐标的最大值（右边界）
        y2 (float): y坐标的最大值（下边界）
        sample_data_token (str): 样本数据令牌，用于唯一标识图像
        filename (str): 对应的图像文件名，标注所在的图像文件

    Returns:
        dict: 一个2D标注记录样本，包含以下字段：
            - file_name (str): 文件名
            - image_id (str): 样本数据令牌（图像ID）
            - area (float): 2D边界框面积
            - category_name (str): 类别名称
            - category_id (int): 类别ID
            - bbox (list[float]): 2D边界框 [左x, 上y, 宽度, 高度]
            - iscrowd (int): 是否为拥挤区域（0表示否）
    """
    # 初始化重投影记录字典，使用OrderedDict保持字段顺序
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    
    # 初始化COCO格式记录字典
    coco_rec = dict()

    # 定义需要从原始标注中提取的相关字段
    # 这些字段包含了物体的重要属性和关联信息
    relevant_keys = [
        'attribute_tokens',        # 属性令牌列表（如运动状态等）
        'category_name',          # 类别名称（如car, pedestrian等）
        'instance_token',         # 实例令牌，用于跨帧跟踪同一物体
        'next',                   # 下一帧中对应标注的令牌
        'num_lidar_pts',         # 激光雷达点云中的点数
        'num_radar_pts',         # 雷达点云中的点数
        'prev',                  # 前一帧中对应标注的令牌
        'sample_annotation_token', # 样本标注令牌
        'sample_data_token',      # 样本数据令牌
        'visibility_token',       # 可见性令牌（表示遮挡程度）
    ]

    # 从原始标注记录中提取相关字段
    # 只保留我们需要的字段，过滤掉不必要的信息
    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    # 添加2D边界框角点坐标
    # 格式：[左, 上, 右, 下]
    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    
    # 添加图像文件名
    repro_rec['filename'] = filename

    # 生成COCO格式的基本信息
    coco_rec['file_name'] = filename           # 图像文件名
    coco_rec['image_id'] = sample_data_token   # 图像ID（使用样本数据令牌）
    
    # 计算2D边界框的面积
    # 面积 = 宽度 × 高度
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    # 进行类别名称映射和验证
    # nuScenes使用的类别名称需要映射到标准的检测类别
    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        # 如果类别名称不在映射表中，返回None（过滤掉不支持的类别）
        return None
    
    # 获取映射后的标准类别名称
    # 例如：'vehicle.car' -> 'car'
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    
    # 设置COCO格式的类别信息
    coco_rec['category_name'] = cat_name                    # 标准类别名称
    coco_rec['category_id'] = nus_categories.index(cat_name) # 类别ID（数字索引）
    
    # 设置COCO格式的边界框
    # COCO格式：[左上角x, 左上角y, 宽度, 高度]
    # 从角点坐标转换为COCO格式
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    
    # 设置拥挤标志
    # iscrowd=0 表示这是一个独立的物体实例，不是拥挤区域
    # 在目标检测中，拥挤区域通常用分割掩码而不是边界框表示
    coco_rec['iscrowd'] = 0

    # 返回COCO格式的标注记录
    return coco_rec
