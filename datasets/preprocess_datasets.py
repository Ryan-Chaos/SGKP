import os
import cv2
import argparse

from PIL import Image
from tqdm import tqdm

import numpy as np

import h5py

import torch

import sys
sys.path.append('..')
# from dog import Dog
from pspnet import PSPNet
from lib.utils import *
from lib.photometric import photometricAug
from lib.homographic import homographicAug

from pathlib import Path

# orb_path = Path(__file__).parent / "../extractors/orbslam2_features/lib"
# sys.path.append(str(orb_path))
# from orbslam2_features import ORBextractor

superpoint_path = Path(__file__).parent / "../extractors/SuperPointPretrainedNetwork"
sys.path.append(str(superpoint_path))
from demo_superpoint import SuperPointFrontend

# alike_path = Path(__file__).parent / "../extractors/ALIKE"
# sys.path.append(str(alike_path))
# import alike
# from alike import ALike

config = {
    'augmentation': {             
        'photometric': {
            'enable': True,
            'params': {
                'random_brightness': {
                    'max_abs_change': 50
                },
                'random_contrast': {
                    'strength_range': [0.5, 1.5]
                },
                'additive_gaussian_noise': {
                    'stddev_range': [0, 10]
                },
                'additive_shade': {
                    'max_scale': 0.8,
                    'kernel_size_range': [100, 150]
                },
                'motion_blur': {
                    'max_kernel_size': 3
                },
                'gamma_correction': {
                    'strength_range': [0.5, 2.0]
                }
            }
        },
        'homographic': {
            'enable': True,
            'params': {
                'translation': True,
                'rotation': True,
                'scaling': True,
                'perspective': True,
                'scaling_amplitude': 0.5,
                'perspective_amplitude_x': 0.2,
                'perspective_amplitude_y': 0.2,
                'patch_ratio_range': [0.7, 1.0],
                'max_angle': 3.14,
                'shift': -1,
                'allow_artifacts': True,
            },
            'valid_border_margin': 3
        }
    }
}

def parse_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser(description="The script for pre-extract features from Megadepth")

    parser.add_argument(
        '--descriptor', type=str, default='superpoint',
        help='type of descriptor'
    )

    parser.add_argument(
        '--dataset_name', type=str, default='change',
        help='coco or megadepth'
    )

    parser.add_argument(
        '--dataset_path', type=str, default='D:/work/data/SECOND_train_set/im1',
        help='path to the dataset'
    )

    parser.add_argument(
        '--scene_info_path', type=str, required=False,
        help='path to the processed scenes (only use for MegaDepth)'
    )

    parser.add_argument(
        '--data_type', type=str, default='train',
        help='train or val'
    )

    parser.add_argument(
        '--output_path', type=str, default='D:/work/data/SECOND_train_set/output',
        help='path for saving output'
    )

    parser.add_argument(
        '--num_kps', type=int, default=2048,
        help='number of keypoint to extract'
    )

    parser.add_argument(
        '--matches_ratio', type=float, default=0.025,
        help='matches / keypoints'
    )

    parser.add_argument(
        '--gpu_id', type=str, default='0',
        help='id(s) for CUDA_VISIBLE_DEVICES'
    )

    args = parser.parse_args()
    print(args)

    return args


def extract(feature, extractor, image, num_kps, is_gray=False):
    if 'alike' == feature.lower(): # feature.lower()指的是将feature的值变为小写
        if is_gray:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = extractor(rgb, sub_pixel=True)
        keypoints = pred['keypoints']
        if keypoints.shape[0] <= 1:
            raise EmptyTensorError
        descriptors = pred['descriptors']
        scores = pred['scores']
        keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
    else:
        if not is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 如果图像不是灰度图，将其转换为灰度图
        if 'superpoint' == feature.lower(): # 将图像转换为浮点数并归一化，然后运行提取器
            image = (image.astype('float32') / 255.) # 行代码将图像数据类型转换为32位浮点数，并将像素值归一化到0到1的范围内
            # TODO 修改这里的run函数
            keypoints, descriptors, _ = extractor.run(image) # 调用SuperPoint提取器的run方法来处理归一化后的图像
            if keypoints.shape[1] <= 1:
                raise EmptyTensorError # 检查返回的关键点是否有效。如果关键点的数量小于或等于1，那么没有足够的关键点来进行匹配，因此抛出EmptyTensorError异常
            keypoints, descriptors = keypoints.T, descriptors.T # 将关键点和描述符矩阵转置
        elif feature.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
            image = (image.astype('float32') / 255.)
            keypoints, scores, descriptors = extractor.detectAndCompute(image)
            if keypoints.shape[0] <= 1:
                raise EmptyTensorError
        elif 'orb' == feature.lower():
            kps_tuples, descriptors = extractor.detectAndCompute(image)
            # convert keypoints 
            keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
            if len(keypoints) <= 1:
                raise EmptyTensorError
            keypoints = np.array(
                [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints], 
                dtype=np.float32
            )

    if keypoints.shape[0] > num_kps: # 如果提取的关键点数量超过了num_kps参数指定的数量，将它们截断到该数量
        keypoints, descriptors = keypoints[:num_kps], descriptors[:num_kps]

    return keypoints.astype(np.float32), descriptors # 返回处理后的关键点和描述符


def extract_change(feature, extractor, image, weights, num_kps, is_gray=False):
    if 'alike' == feature.lower():
        if is_gray:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = extractor(rgb, sub_pixel=True)
        keypoints = pred['keypoints']
        if keypoints.shape[0] <= 1:
            raise EmptyTensorError
        descriptors = pred['descriptors']
        scores = pred['scores']
        keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
    else:
        if not is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 如果图像不是灰度图，将其转换为灰度图
        if 'superpoint' == feature.lower(): # 将图像转换为浮点数并归一化，然后运行提取器
            image = (image.astype('float32') / 255.) # 行代码将图像数据类型转换为32位浮点数，并将像素值归一化到0到1的范围内

            # TODO 修改这里的run函数
            keypoints, descriptors, _ = extractor.run_change(image,weights) # 调用SuperPoint提取器的run方法来处理归一化后的图像
            # keypoints2, descriptors2, _ = extractor.run(image2, weights2)
            if keypoints.shape[1] <= 1:
                raise EmptyTensorError # 检查返回的关键点是否有效。如果关键点的数量小于或等于1，那么没有足够的关键点来进行匹配，因此抛出EmptyTensorError异常
            keypoints, descriptors = keypoints.T, descriptors.T # 将关键点和描述符矩阵转置
        elif feature.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
            image = (image.astype('float32') / 255.)
            keypoints, scores, descriptors = extractor.detectAndCompute(image)
            if keypoints.shape[0] <= 1:
                raise EmptyTensorError
        elif 'orb' == feature.lower():
            kps_tuples, descriptors = extractor.detectAndCompute(image)
            # convert keypoints
            keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
            if len(keypoints) <= 1:
                raise EmptyTensorError
            keypoints = np.array(
                [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints],
                dtype=np.float32
            )

    if keypoints.shape[0] > num_kps: # 如果提取的关键点数量超过了num_kps参数指定的数量，将它们截断到该数量
        keypoints, descriptors = keypoints[:num_kps], descriptors[:num_kps]

    return keypoints.astype(np.float32), descriptors # 返回处理后的关键点和描述符


def check_coco(kps1, kps2, homography1, homography2, matches_ratio):
    """
    用于验证两组关键点（keypoints）之间的对应关系是否满足单应性（homography）变换的条件，并且是否达到了预设的匹配比例
    """
    pos_radius = 3
    kps1_pos = torch.from_numpy(kps1[:, :2].T).cuda()
    kps2_pos = torch.from_numpy(kps2[:, :2].T).cuda()
    homo1 = torch.from_numpy(homography1.astype(np.float32)).view(3, 3).cuda()
    homo2 = torch.from_numpy(homography2.astype(np.float32)).view(3, 3).cuda()

    # Find kps1 correspondences
    kps1_warp_pos = warpPerspective(kps1_pos, homo1)
    pos_dist1 = torch.max(
        torch.abs(
            kps1_warp_pos.unsqueeze(2).float() -
            kps2_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist1 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps1.shape[0]) < matches_ratio:
        raise EmptyTensorError

    # Find kps2 correspondences
    kps2_warp_pos = warpPerspective(kps2_pos, homo2)
    pos_dist2 = torch.max(
        torch.abs(
            kps2_warp_pos.unsqueeze(2).float() -
            kps1_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist2 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps2.shape[0]) < matches_ratio:
        raise EmptyTensorError
    
    return pos_dist1.cpu().numpy(), pos_dist2.cpu().numpy()


def process_coco(feature, extractor, dataset_path, output_path, num_kps, matches_ratio):
    img_list = os.listdir(dataset_path)
    count = 0
    for img_name in img_list:
        image_path = os.path.join(dataset_path, img_name)

        image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image1_ = image1.copy()

        # 特征提取:
        # 使用extract函数从图像中提取关键点kps1和描述符descs1
        try:
            kps1, descs1 = extract(feature, extractor, image1, num_kps, True)
        except EmptyTensorError:
            continue
        # 使用normalize_keypoints函数将关键点坐标归一化
        normalized_kps1 = normalize_keypoints(kps1, image1.shape)

        # apply homography transformation to image 使用homographicAug和photometricAug进行同态变换和光度增强
        homoAugmentor = homographicAug(**config['augmentation']['homographic'])
        photoAugmentor = photometricAug(**config['augmentation']['photometric'])

        # 生成变换后的图像
        image2 = photoAugmentor(image1_)
        image2 = torch.tensor(image2, dtype=torch.float32) / 255.
        image2, homography1, homography2, valid_mask = homoAugmentor(image2)
        image2 = (image2.numpy() * 255.).astype(np.uint8)

        # 从变换后的图像image2中提取关键点kps2和描述符descs2
        try:
            kps2, descs2 = extract(feature, extractor, image2, num_kps, True)
        except EmptyTensorError:
            continue
        normalized_kps2 = normalize_keypoints(kps2, image2.shape)

        # 使用check_coco函数检查两组关键点之间的匹配距离
        try:
            pos_dist1, pos_dist2 = check_coco(kps1, kps2, homography1, homography2, matches_ratio)
        except EmptyTensorError:
            continue

        # TODO 将关键点、描述符和匹配距离保存到.npz文件中
        save_path = os.path.join(output_path, str(count) + '.npz')
        with open(save_path, 'wb') as file:
            np.savez(
                file,
                kps1=kps1,
                normalized_kps1=normalized_kps1,
                descs1=descs1,
                pos_dist1=pos_dist1.astype(np.float16),
                ids1=np.arange(0, kps1.shape[0]),
                kps2=kps2,
                normalized_kps2=normalized_kps2,
                descs2=descs2,
                pos_dist2=pos_dist2.astype(np.float16),
                ids2=np.arange(0, kps2.shape[0])
            )

        count += 1
        if count == 10000:
            break
    print("Sampled %d image pairs for COCO" % count)

def process_change(feature, extractor, dataset_path, output_path, num_kps, matches_ratio):
    img_list = os.listdir(dataset_path) # path写到im1
    count = 0
    for img_name in tqdm(img_list):
        image_path1 = os.path.join(dataset_path, img_name)
        image_path2 = image_path1.replace("im1", "im2")

        image1_rgb = Image.open(image_path1)
        image2_rgb = Image.open(image_path2)

        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        # image1_ = image1.copy()

        segmodel = PSPNet()
        seg_result1 = segmodel.detect_image(image1_rgb, count=False)
        seg_result2 = segmodel.detect_image(image2_rgb, count=False)

        # 取得权重
        weights1, weights2 = segmantic_to_confidence_static(seg_result1, seg_result2)
        # weights1, weights2 = segmantic_to_confidence(seg_result1, seg_result2)

        # 特征提取:
        # 使用extract函数从图像中提取关键点kps1和描述符descs1
        # TODO 修改：输入两张图像，提取特征点和描述符，将类别标签作为参数传进去（现算也行），最终修改的是spp中的热力图（置信度）
        try:
            kps1, descs1 = extract_change(feature, extractor, image1, weights1, num_kps, True)
        except EmptyTensorError:
            continue
        # 使用normalize_keypoints函数将关键点坐标归一化
        normalized_kps1 = normalize_keypoints(kps1, image1.shape)

        # apply homography transformation to image 使用homographicAug和photometricAug进行同态变换和光度增强
        homoAugmentor = homographicAug(**config['augmentation']['homographic'])
        photoAugmentor = photometricAug(**config['augmentation']['photometric'])

        # 生成变换后的图像
        image2 = photoAugmentor(image2)
        image2 = torch.tensor(image2, dtype=torch.float32) / 255.
        image2, homography1, homography2, valid_mask = homoAugmentor(image2)
        image2 = (image2.numpy() * 255.).astype(np.uint8)

        # 从变换后的图像image2中提取关键点kps2和描述符descs2
        try:
            kps2, descs2 = extract_change(feature, extractor, image2, weights2, num_kps, True)
        except EmptyTensorError:
            continue
        normalized_kps2 = normalize_keypoints(kps2, image2.shape)

        # 使用check_coco函数检查两组关键点之间的匹配距离
        try:
            pos_dist1, pos_dist2 = check_coco(kps1, kps2, homography1, homography2, matches_ratio)
        except EmptyTensorError:
            continue

        # TODO 将关键点、描述符和匹配距离保存到.npz文件中
        save_path = os.path.join(output_path, str(count) + '.npz')
        with open(save_path, 'wb') as file:
            np.savez(
                file,
                kps1=kps1,
                normalized_kps1=normalized_kps1,
                descs1=descs1,
                pos_dist1=pos_dist1.astype(np.float16),
                ids1=np.arange(0, kps1.shape[0]),
                kps2=kps2,
                normalized_kps2=normalized_kps2,
                descs2=descs2,
                pos_dist2=pos_dist2.astype(np.float16),
                ids2=np.arange(0, kps2.shape[0])
            )

        count += 1
        if count == 10000:
            break
    print("Sampled %d image pairs for Change" % count)


def process_static_change(feature, extractor, dataset_path, output_path, num_kps, matches_ratio):
    img_list = os.listdir(dataset_path)
    count = 0
    for img_name in tqdm(img_list):
        image_path1 = os.path.join(dataset_path, img_name)
        image_path2 = image_path1.replace("im1", "im2")

        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)


        # 特征提取:
        # 使用extract函数从图像中提取关键点kps1和描述符descs1
        try:
            kps1, descs1 = extract(feature, extractor, image1, num_kps, True)
        except EmptyTensorError:
            continue
        # 使用normalize_keypoints函数将关键点坐标归一化
        normalized_kps1 = normalize_keypoints(kps1, image1.shape)

        # apply homography transformation to image 使用homographicAug和photometricAug进行同态变换和光度增强
        homoAugmentor = homographicAug(**config['augmentation']['homographic'])
        photoAugmentor = photometricAug(**config['augmentation']['photometric'])

        # 生成变换后的图像
        image2 = photoAugmentor(image2)
        image2 = torch.tensor(image2, dtype=torch.float32) / 255.
        image2, homography1, homography2, valid_mask = homoAugmentor(image2)
        image2 = (image2.numpy() * 255.).astype(np.uint8)

        # 从变换后的图像image2中提取关键点kps2和描述符descs2
        try:
            kps2, descs2 = extract(feature, extractor, image2, num_kps, True)
        except EmptyTensorError:
            continue
        normalized_kps2 = normalize_keypoints(kps2, image2.shape)

        # 使用check_coco函数检查两组关键点之间的匹配距离
        try:
            pos_dist1, pos_dist2 = check_coco(kps1, kps2, homography1, homography2, matches_ratio)
        except EmptyTensorError:
            continue

        # TODO 将关键点、描述符和匹配距离保存到.npz文件中
        save_path = os.path.join(output_path, str(count) + '.npz')
        with open(save_path, 'wb') as file:
            np.savez(
                file,
                kps1=kps1,
                normalized_kps1=normalized_kps1,
                descs1=descs1,
                pos_dist1=pos_dist1.astype(np.float16),
                ids1=np.arange(0, kps1.shape[0]),
                kps2=kps2,
                normalized_kps2=normalized_kps2,
                descs2=descs2,
                pos_dist2=pos_dist2.astype(np.float16),
                ids2=np.arange(0, kps2.shape[0])
            )

        count += 1
        if count == 10000:
            break
    print("Sampled %d image pairs for static_change" % count)

def segmantic_to_confidence(img1, img2): # 注意，这里输入的img1和img2是标签图，非原图
    """
    convert semantic map to confidence for other usage
    """

    common_pixels = np.intersect1d(img1, img2)

    similarities = {}
    img1_tensor = torch.from_numpy(img1).to('cuda')
    out1 = torch.zeros_like(img1_tensor).float() # 创建一个与seg_map形状相同的全0张量
    out2 = torch.zeros_like(img1_tensor).float()


    # 对每个共有的像素值计算相似度
    for pixel_value in common_pixels:
        # 提取img1和img2中当前像素值的区域
        region1 = np.where(img1 == pixel_value, 1, 0)
        region2 = np.where(img2 == pixel_value, 1, 0)

        # 计算两个区域之间的相似度
        similarity = normalized_cross_correlation(region1,region2)

        # similarities.append(similarity)
        similarities[pixel_value] = similarity


    # 将相似度按降序排序
    # sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}

    # print(sorted_similarities)

    # 将排名前三的像素值在两个图像中的对应部分替换为1
    num_common_pixels = len(sorted_similarities)
    # print(np.unique(out1))
    for i, (pixel_value, similarity) in enumerate(sorted_similarities.items()):
        # TODO 如何划分权重值，做实验吧
        if i < num_common_pixels * 1 / 2 and similarity > 0:
        # if similarity > 0:
        # if i == 1:
            out1[img1 == pixel_value] = 1.0
            out2[img2 == pixel_value] = 1.0
            # print(pixel_value,"为1")
            # print(np.unique(out1))
        else:
            out1[img1 == pixel_value] = 0.5
            out2[img2 == pixel_value] = 0.5
            # print(pixel_value, "为0.5")
            # print(np.unique(out1))

    all_pixels = np.unique(np.concatenate((img1, img2)))
    # print("all:",all_pixels)
    for pixel_value in all_pixels:
        if pixel_value not in sorted_similarities:
            out1[out1 == pixel_value] = 0.1
            out2[out2 == pixel_value] = 0.1

    # 将PyTorch张量转换为NumPy数组
    out1 = out1.cpu().numpy()
    out2 = out2.cpu().numpy()

    return out1,out2

def segmantic_to_confidence_static(img1, img2): # 注意，这里输入的img1和img2是标签图，非原图
    """
    convert semantic map to confidence for other usage
    """

    common_pixels = np.intersect1d(img1, img2)

    similarities = {}
    img1_tensor = torch.from_numpy(img1).to('cuda')
    out1 = torch.zeros_like(img1_tensor).float() # 创建一个与seg_map形状相同的全0张量
    out2 = torch.zeros_like(img1_tensor).float()


    # 对每个共有的像素值计算相似度
    for pixel_value in common_pixels:
        # 提取img1和img2中当前像素值的区域
        region1 = np.where(img1 == pixel_value, 1, 0)
        region2 = np.where(img2 == pixel_value, 1, 0)

        # 计算两个区域之间的相似度
        similarity = normalized_cross_correlation(region1,region2)

        # similarities.append(similarity)
        similarities[pixel_value] = similarity


    # 将相似度按降序排序
    # sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}

    all_pixels = np.unique(np.concatenate((img1, img2)))
    # print("all:",all_pixels)
    for pixel_value in all_pixels:
        if pixel_value not in sorted_similarities:
            out1[img1 == pixel_value] = 0.1
            out2[img2 == pixel_value] = 0.1
        else:
            out1[img1 == pixel_value] = 1.0
            out2[img2 == pixel_value] = 1.0

    # 将PyTorch张量转换为NumPy数组
    out1 = out1.cpu().numpy()
    out2 = out2.cpu().numpy()

    return out1,out2


def normalized_cross_correlation(img1, img2):
    # 计算相关系数，这里使用的是有偏估计
    correlation = np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2)+1e-10)

    return correlation



def check_mega(kps1, kps2, base_path, pair_metadata, matches_ratio):
    pos_radius = 3

    # convert u, v to i, j
    kps1_pos_ = kps1[:,:2]
    kps1_pos_ = kps1_pos_[:, ::-1].copy()
    kps2_pos_ = kps2[:,:2]
    kps2_pos_ = kps2_pos_[:, ::-1].copy()
    kps1_pos = torch.from_numpy(kps1_pos_.T).cuda()
    kps2_pos = torch.from_numpy(kps2_pos_.T).cuda()

    # depth, intrinsics, pose
    depth_path1 = os.path.join(
        base_path, pair_metadata['depth_path1']
    )
    with h5py.File(depth_path1, 'r') as hdf5_file:
        depth1 = np.array(hdf5_file['/depth'])
    assert(np.min(depth1) >= 0)
    depth1 = torch.from_numpy(depth1.astype(np.float32)).cuda()
    intrinsics1 = pair_metadata['intrinsics1']
    intrinsics1 = torch.from_numpy(intrinsics1.astype(np.float32)).cuda()
    pose1 = pair_metadata['pose1']
    pose1 = torch.from_numpy(pose1.astype(np.float32)).cuda()

    depth_path2 = os.path.join(
        base_path, pair_metadata['depth_path2']
    )
    with h5py.File(depth_path2, 'r') as hdf5_file:
        depth2 = np.array(hdf5_file['/depth'])
    assert(np.min(depth2) >= 0)
    depth2 = torch.from_numpy(depth2.astype(np.float32)).cuda()
    intrinsics2 = pair_metadata['intrinsics2']
    intrinsics2 = torch.from_numpy(intrinsics2.astype(np.float32)).cuda()
    pose2 = pair_metadata['pose2']
    pose2 = torch.from_numpy(pose2.astype(np.float32)).cuda()

    # Find kps1 correspondences
    kps1_pos, kps1_warp_pos, ids1 = warp(
        kps1_pos,
        depth1, intrinsics1, pose1, (0.0, 0.0),
        depth2, intrinsics2, pose2, (0.0, 0.0)
    )
    pos_dist1 = torch.max(
        torch.abs(
            kps1_warp_pos.unsqueeze(2).float() -
            kps2_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist1 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps1.shape[0]) < matches_ratio:
        # print(ids_has_gt.sum(), kps1.shape[0])
        raise EmptyTensorError

    # Find kps2 correspondences
    kps2_pos, kps2_warp_pos, ids2 = warp(
        kps2_pos,
        depth2, intrinsics2, pose2, (0.0, 0.0),
        depth1, intrinsics1, pose1, (0.0, 0.0)
    )
    kps1_pos = torch.from_numpy(kps1_pos_.T).cuda()
    pos_dist2 = torch.max(
        torch.abs(
            kps2_warp_pos.unsqueeze(2).float() -
            kps1_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist2 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps2.shape[0]) < matches_ratio:
        raise EmptyTensorError
    
    return pos_dist1.cpu().numpy(), pos_dist2.cpu().numpy(), ids1.cpu().numpy(), ids2.cpu().numpy()


def process_megadepth(feature, extractor, dataset_path, scene_info_path, output_path, num_kps, matches_ratio, train):
    scenes = []
    if train:
        scene_list_path='megadepth_utils/train_scenes.txt'
    else:
        scene_list_path='megadepth_utils/valid_scenes.txt',
    with open(scene_list_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            scenes.append(line.strip('\n'))
    
    total_count = 0
    for scene in tqdm(scenes, total=len(scenes)):
        scene_path = os.path.join(
            scene_info_path, '%s.npz' % scene
        )
        if not os.path.exists(scene_path):
            continue
        scene_info = np.load(scene_path, allow_pickle=True)
        overlap_matrix = scene_info['overlap_matrix']
        valid = np.logical_and(
                    overlap_matrix >= 0.1,
                    overlap_matrix <= 1.0
                )
        pairs = np.vstack(np.where(valid))
        pairs_idx = [i for i in range(pairs.shape[1])]
        np.random.shuffle(pairs_idx)
        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']
        intrinsics = scene_info['intrinsics']
        poses = scene_info['poses']
        
        count = 0
        for pair_idx in pairs_idx:
            idx1 = pairs[0, pair_idx]
            image_path1 = os.path.join(dataset_path, image_paths[idx1])
            idx2 = pairs[1, pair_idx]
            image_path2 = os.path.join(dataset_path, image_paths[idx2])

            image1 = cv2.imread(image_path1)
            try:
                kps1, descs1 = extract(feature, extractor, image1, num_kps)
            except EmptyTensorError:
                continue
            normalized_kps1 = normalize_keypoints(kps1, image1.shape)
            image2 = cv2.imread(image_path2)
            try:
                kps2, descs2 = extract(feature, extractor, image2, num_kps)
            except EmptyTensorError:
                continue
            normalized_kps2 = normalize_keypoints(kps2, image2.shape)

            pair_data = {
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2]
            }
            try:
                pos_dist1, pos_dist2, ids1, ids2 = check_mega(kps1, kps2, dataset_path, pair_data, matches_ratio)
            except EmptyTensorError:
                continue

            save_path = os.path.join(output_path, str(total_count + count) + '.npz')
            with open(save_path, 'wb') as file:
                np.savez(
                    file,
                    kps1=kps1,
                    normalized_kps1=normalized_kps1,
                    descs1=descs1,
                    pos_dist1=pos_dist1.astype(np.float16),
                    ids1=ids1,
                    kps2=kps2,
                    normalized_kps2=normalized_kps2,
                    descs2=descs2,
                    pos_dist2=pos_dist2.astype(np.float16),
                    ids2=ids2
            )

            count += 1
            if count == 500:
                break
        
        total_count += count

    print("Sampled %d image pairs for MegaDepth" % total_count)




if __name__ == '__main__':
    # command line arguments
    args = parse_arguments()

    # CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("Using CUDA!!!")

    # check dataset 数据集检查:
    # 检查args.dataset_name是否为支持的数据集
    if args.dataset_name.lower() not in ['coco', 'megadepth', 'change']:
        raise Exception('Not supported datatse: "%s".' % args.dataset_name)

    feature = args.descriptor

    # define the feature extractor 特征提取器定义:
    # 根据args.descriptor选择不同的特征提取器。支持的提取器包括SIFT、RootSIFT、SOSNet、HardNet、ORB、SuperPoint和ALike
    if args.descriptor.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
        # extractor = Dog(descriptor=args.descriptor.lower())
        print("descriptor的值为['sift', 'rootsift', 'sosnet', 'hardnet']")
    # elif 'orb' == args.descriptor.lower():
    #     extractor = ORBextractor(3000, 1.2, 8)
    elif 'superpoint' == args.descriptor.lower():
        sp_weights_path = Path(__file__).parent / "../extractors/SuperPointPretrainedNetwork/superpoint_v1.pth"
        extractor = SuperPointFrontend(weights_path=sp_weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=use_cuda)
    # elif 'alike' == args.descriptor.lower():
    #     extractor = ALike(**alike.configs['alike-l'], device='cuda' if use_cuda else 'cpu', top_k=-1, scores_th=0.2)
    else:
        raise Exception('Not supported descriptor: "%s".' % args.descriptor)

    # output path
    if not os.path.isdir(os.path.join(args.output_path, args.data_type)):
        os.mkdir(os.path.join(args.output_path, args.data_type))
    output_path = os.path.join(args.output_path, args.data_type, args.descriptor)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # else:
    #     print("Found existing folder! Please check out!!!")
    #     exit(-1)

    # 数据处理:
    # 根据args.dataset_name处理COCO或MegaDepth数据集。
    # process_coco 和 process_megadepth 函数用于处理相应的数据集。
    if args.dataset_name.lower() == 'coco':
        print("Processing COCO...")
        dataset_path = os.path.join(args.dataset_path, args.data_type+'2014')
        process_coco(feature, extractor, dataset_path, output_path, args.num_kps, args.matches_ratio)
    elif args.dataset_name.lower() == 'change':
        print("Processing ChangeDection...")
        dataset_path = os.path.join(args.dataset_path)
        process_change(feature, extractor, dataset_path, output_path, args.num_kps, args.matches_ratio)
        # process_static_change(feature, extractor, dataset_path, output_path, args.num_kps, args.matches_ratio)
    else:
        print("Processing MegaDepth...")
        process_megadepth(feature, extractor, args.dataset_path, args.scene_info_path, output_path, args.num_kps, args.matches_ratio, train = True if args.data_type == 'train' else False)

    print('Done!')