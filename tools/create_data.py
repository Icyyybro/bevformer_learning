# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

# 导入必要的模块
from data_converter.create_gt_database import create_groundtruth_database  # 创建真值数据库的函数
from data_converter import nuscenes_converter as nuscenes_converter  # nuScenes数据集转换器
from data_converter import lyft_converter as lyft_converter  # Lyft数据集转换器
from data_converter import kitti_converter as kitti  # KITTI数据集转换器
from data_converter import indoor_converter as indoor  # 室内数据集转换器
import argparse  # 命令行参数解析
from os import path as osp  # 路径操作工具
import sys
sys.path.append('.')  # 将当前目录添加到Python路径中


def kitti_data_prep(root_path, info_prefix, version, out_dir):
    """准备KITTI数据集相关数据的函数
    
    该函数会生成包含基本信息的.pkl文件、2D标注和真值数据库
    
    Args:
        root_path (str): 数据集根目录路径
        info_prefix (str): 信息文件名的前缀
        version (str): 数据集版本
        out_dir (str): 真值数据库信息的输出目录
    """
    # 创建KITTI信息文件，包含点云、图像、标注等基本信息
    kitti.create_kitti_info_file(root_path, info_prefix)
    # 创建降采样的点云数据，减少数据量提高处理效率
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    # 构建各个数据集分割的信息文件路径
    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')  # 训练集信息文件路径
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')  # 验证集信息文件路径
    info_trainval_path = osp.join(root_path, f'{info_prefix}_infos_trainval.pkl')  # 训练+验证集信息文件路径
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')  # 测试集信息文件路径
    
    # 为各个数据集分割导出2D标注信息
    kitti.export_2d_annotation(root_path, info_train_path)  # 导出训练集2D标注
    kitti.export_2d_annotation(root_path, info_val_path)  # 导出验证集2D标注
    kitti.export_2d_annotation(root_path, info_trainval_path)  # 导出训练+验证集2D标注
    kitti.export_2d_annotation(root_path, info_test_path)  # 导出测试集2D标注

    # 创建真值数据库，用于数据增强等操作
    create_groundtruth_database(
        'KittiDataset',  # 数据集类名
        root_path,  # 数据集根路径
        info_prefix,  # 信息文件前缀
        f'{out_dir}/{info_prefix}_infos_train.pkl',  # 训练集信息文件路径
        relative_path=False,  # 不使用相对路径
        mask_anno_path='instances_train.json',  # 掩码标注文件路径
        with_mask=(version == 'mask'))  # 是否包含掩码信息


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """准备nuScenes数据集相关数据的函数
    
    该函数会生成包含基本信息的.pkl文件、2D标注和真值数据库
    
    Args:
        root_path (str): 数据集根目录路径
        can_bus_root_path (str): CAN总线数据根目录路径
        info_prefix (str): 信息文件名的前缀
        version (str): 数据集版本
        dataset_name (str): 数据集类名
        out_dir (str): 真值数据库信息的输出目录
        max_sweeps (int): 输入连续帧的数量，默认为10
    """
    # 创建nuScenes信息文件，包含多帧时序信息
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    # 根据数据集版本处理不同的数据分割
    if version == 'v1.0-test':
        # 如果是测试版本，只处理测试集
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
    else:
        # 如果是训练版本，处理训练集和验证集
        info_train_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_train.pkl')  # 训练集时序信息文件
        info_val_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_val.pkl')  # 验证集时序信息文件
        
        # 导出训练集和验证集的2D标注
        nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)
        
        # 注释掉的代码：创建真值数据库（在nuScenes中通常不需要）
        # create_groundtruth_database(dataset_name, root_path, info_prefix,
        #                             f'{out_dir}/{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """准备Lyft数据集相关数据的函数
    
    该函数会生成包含基本信息的.pkl文件
    虽然Lyft数据集中不使用真值数据库和2D标注，但也可以像nuScenes一样生成
    
    Args:
        root_path (str): 数据集根目录路径
        info_prefix (str): 信息文件名的前缀
        version (str): 数据集版本
        max_sweeps (int, optional): 输入连续帧的数量，默认为10
    """
    # 创建Lyft信息文件
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """准备ScanNet数据集的信息文件
    
    Args:
        root_path (str): 数据集根目录路径
        info_prefix (str): 信息文件名的前缀
        out_dir (str): 生成信息文件的输出目录
        workers (int): 使用的线程数量
    """
    # 创建室内数据集（ScanNet）的信息文件
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """准备S3DIS数据集的信息文件
    
    Args:
        root_path (str): 数据集根目录路径
        info_prefix (str): 信息文件名的前缀
        out_dir (str): 生成信息文件的输出目录
        workers (int): 使用的线程数量
    """
    # 创建室内数据集（S3DIS）的信息文件
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """准备SUN RGB-D数据集的信息文件
    
    Args:
        root_path (str): 数据集根目录路径
        info_prefix (str): 信息文件名的前缀
        out_dir (str): 生成信息文件的输出目录
        workers (int): 使用的线程数量
    """
    # 创建室内数据集（SUN RGB-D）的信息文件
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """准备Waymo数据集的信息文件
    
    Args:
        root_path (str): 数据集根目录路径
        info_prefix (str): 信息文件名的前缀
        out_dir (str): 生成信息文件的输出目录
        workers (int): 使用的线程数量
        max_sweeps (int): 输入连续帧的数量，默认为5
            这里我们存储这些帧的位姿信息以供后续使用
    """
    # 导入Waymo转换器
    from tools.data_converter import waymo_converter as waymo

    # 定义数据集分割
    splits = ['training', 'validation', 'testing']

    # 遍历每个数据集分割，将Waymo格式转换为KITTI格式
    for i, split in enumerate(splits):
        # 构建加载目录路径（Waymo原始格式）
        load_dir = osp.join(root_path, 'waymo_format', split)
        
        # 构建保存目录路径（KITTI格式）
        if split == 'validation':
            # 验证集保存到训练目录（Waymo的验证集作为KITTI的训练集）
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        
        # 创建Waymo到KITTI的转换器
        converter = waymo.Waymo2KITTI(
            load_dir,  # 源目录
            save_dir,  # 目标目录
            prefix=str(i),  # 文件前缀
            workers=workers,  # 工作线程数
            test_mode=(split == 'test'))  # 是否为测试模式
        
        # 执行转换
        converter.convert()
    
    # 更新输出目录为KITTI格式目录
    out_dir = osp.join(out_dir, 'kitti_format')
    
    # 生成Waymo信息文件（KITTI格式）
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)

    # 创建真值数据库
    create_groundtruth_database(
        'WaymoDataset',  # 数据集类名
        out_dir,  # 输出目录
        info_prefix,  # 信息文件前缀
        f'{out_dir}/{info_prefix}_infos_train.pkl',  # 训练集信息文件路径
        relative_path=False,  # 不使用相对路径
        with_mask=False)  # 不包含掩码信息


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='数据转换器参数解析器')

# 添加各种命令行参数
parser.add_argument('dataset', metavar='kitti', help='数据集名称')  # 位置参数：数据集名称
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='指定数据集的根路径')  # 数据集根路径
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='指定nuScenes CAN总线数据的根路径')  # nuScenes CAN总线数据路径
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='指定数据集版本，KITTI不需要此参数')  # 数据集版本
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='指定每个样本的激光雷达扫描次数')  # 最大扫描次数
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='信息pkl文件的输出目录')  # 输出目录
parser.add_argument('--extra-tag', type=str, default='kitti', help='额外标签')  # 额外标签
parser.add_argument(
    '--workers', type=int, default=4, help='使用的线程数量')  # 工作线程数

# 解析命令行参数
args = parser.parse_args()

# 主程序入口
if __name__ == '__main__':
    # 根据指定的数据集类型执行相应的数据预处理
    if args.dataset == 'kitti':
        # 处理KITTI数据集
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        # 处理nuScenes完整数据集（非mini版本）
        # 处理训练+验证集
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        
        # 处理测试集
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        # 处理nuScenes mini数据集
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        # 处理Lyft数据集
        # 处理训练集
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        
        # 处理测试集
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        # 处理Waymo数据集
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        # 处理ScanNet数据集
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        # 处理S3DIS数据集
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        # 处理SUN RGB-D数据集
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
