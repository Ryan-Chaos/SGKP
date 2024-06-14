import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch

from datasets.preprocess_datasets import segmantic_to_confidence_static,segmantic_to_confidence
# from dog import Dog
from featurebooster import FeatureBooster

import sys
from pathlib import Path

from pspnet import PSPNet

# orb_path = Path(__file__).parent / "extractors/orbslam2_features/lib"
# sys.path.append(str(orb_path))
# from orbslam2_features import ORBextractor

superpoint_path = Path(__file__).parent / "extractors/SuperPointPretrainedNetwork"
sys.path.append(str(superpoint_path))
from demo_superpoint import SuperPointFrontend

# alike_path = Path(__file__).parent / "extractors/ALIKE"
# sys.path.append(str(alike_path))
# import alike
# from alike import ALike


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract feature and refine descriptor using neural network.")
    
    parser.add_argument(
        '--descriptor', type=str, default='SuperPoint+Boost-F',
        # '--descriptor', type=str, default='SuperPoint',
        help='descriptor to extract'
    )
    
    parser.add_argument(
        '--image_list_file', type=str, default=Path(__file__).parent / "change_detection_val_list.txt",
        help='path to a file containing a list of images to process'
    )

    parser.add_argument(
        '--gpu_id', type=str, default='0',
        help='id(s) for CUDA_VISIBLE_DEVICES'
    )
    
    args = parser.parse_args()

    print(args)

    return args

def normalize_keypoints(keypoints, image_shape):
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    kps = np.array(keypoints)
    kps[:, 0] = (keypoints[:, 0] - x0) / scale
    kps[:, 1] = (keypoints[:, 1] - y0) / scale
    return kps 

if __name__ == '__main__':
    # command line arguments
    args = parse_arguments()
    
    # set CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # set torch grad 禁用梯度计算
    torch.set_grad_enabled(False)

    # 选择不同的特征提取器
    if args.descriptor.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
        # feature_extractor = Dog(descriptor=args.descriptor.lower())
        print("use sift")
    elif 'sift' in args.descriptor.lower():
        print("use sift")
        # feature_extractor = Dog(descriptor='sift')
    elif 'orb' in args.descriptor.lower():
        # feature_extractor = ORBextractor(3000, 1.2, 8)
        print("use orb")
    elif 'superpoint' in args.descriptor.lower():
        sp_weights_path = Path(__file__).parent / "extractors/SuperPointPretrainedNetwork/superpoint_v1.pth"
        feature_extractor = SuperPointFrontend(weights_path=sp_weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=use_cuda)
    elif 'alike' in args.descriptor.lower():
        # feature_extractor = ALike(**alike.configs['alike-l'], device='cuda' if use_cuda else 'cpu', top_k=-1, scores_th=0.2)
        print("use alike")
    else:
        raise Exception('Not supported descriptor: "%s".' % args.descriptor)

    # set FeatureBooster 如果args.descriptor包含+Boost-，则会加载一个配置文件，并创建一个FeatureBooster模型来增强特征描述符
    if "+Boost-" in args.descriptor:
        # load json config file
        config_file = Path(__file__).parent / "config.yaml"
        with open(str(config_file), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(config[args.descriptor])

        # Model
        feature_booster = FeatureBooster(config[args.descriptor])
        if use_cuda:
            feature_booster.cuda()
        feature_booster.eval()
        # load the model 载入模型
        # model_path = Path(__file__).parent / str("models/" + args.descriptor + ".pth")
        model_path = Path(__file__).parent / str("runs/megacoco_k+_SuperPoint+Boost-F_bs16_ep50_lr0.001_adamw_cos_lboost10_warmup500_tune/checkpoints/epoch50.pth")
        # model_path = Path(__file__).parent / str("models/SuperPoint+Boost-F.pth")
        # print(model_path)
        model_dict = torch.load(model_path)
        feature_booster.load_state_dict(model_dict['feature_booster'])
        # feature_booster.load_state_dict(model_dict)

    # Process the file 处理图像文件并提取特征
    with open(args.image_list_file, 'r') as f: # 打开一个包含图像路径列表的文件
        lines = f.readlines()
        half_lines = lines[:len(lines) // 2]

    for line in tqdm(half_lines, total=len(half_lines)): # 遍历每一行（每个图像路径）
        path1 = line.strip()
        path2 = path1.replace('im1', 'im2')

        image1_rgb = Image.open(path1)
        image2_rgb = Image.open(path2)

        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)

        segmodel = PSPNet()
        seg_result1 = segmodel.detect_image(image1_rgb, count=False)
        seg_result2 = segmodel.detect_image(image2_rgb, count=False)

        # 取得权重
        weights1, weights2 = segmantic_to_confidence_static(seg_result1, seg_result2)
        # weights1, weights2 = segmantic_to_confidence(seg_result1, seg_result2)
        # file_name = os.path.basename(path)
        # new_path = os.path.join("D:/study/data/change/output/sppF", file_name)


        if 'alike' in args.descriptor.lower():
            # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # pred = feature_extractor(rgb, sub_pixel=True)
            # keypoints = pred['keypoints']
            # descriptors = pred['descriptors']
            # scores = pred['scores']
            # keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
            print("use alike")
        else:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            if 'superpoint' in args.descriptor.lower():
                image1 = (image1.astype('float32') / 255.)
                image2 = (image2.astype('float32') / 255.)
                keypoints1, descriptors1, _ = feature_extractor.run_change(image1,weights1)
                keypoints2, descriptors2, _ = feature_extractor.run_change(image2,weights2)
                # keypoints1, descriptors1, _ = feature_extractor.run(image1)
                # keypoints2, descriptors2, _ = feature_extractor.run(image2)
                keypoints1, descriptors1 = keypoints1.T, descriptors1.T
                keypoints2, descriptors2 = keypoints2.T, descriptors2.T
            elif args.descriptor.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
                # image = (image.astype('float32') / 255.)
                # keypoints, scores, descriptors = feature_extractor.detectAndCompute(image)
                # keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
                print("use sift")
            elif 'sift' in args.descriptor.lower():
                # image = (image.astype('float32') / 255.)
                print("use sift")
                # keypoints, scores, descriptors = feature_extractor.detectAndCompute(image)
            elif 'orb' in args.descriptor.lower():
                # kps_tuples, descriptors = feature_extractor.detectAndCompute(image)
                # # convert keypoints
                # keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
                # keypoints = np.array(
                #     [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints],
                #     dtype=np.float32
                # )
                print("use orb")

        if "+Boost-" in args.descriptor:
            # boosted the descriptor using trained model
            # 标准化特征点坐标
            kps1 = normalize_keypoints(keypoints1, image1.shape)
            kps1 = torch.from_numpy(kps1.astype(np.float32))
            kps2 = normalize_keypoints(keypoints2, image2.shape)
            kps2 = torch.from_numpy(kps2.astype(np.float32))
            if 'orb' in args.descriptor.lower():
                # descriptors = np.unpackbits(descriptors, axis=1, bitorder='little')
                # descriptors = descriptors * 2.0 - 1.0
                print("use orb")
            descriptors1 = torch.from_numpy(descriptors1.astype(np.float32))
            descriptors2 = torch.from_numpy(descriptors2.astype(np.float32))
            if use_cuda:
                kps1 = kps1.cuda()
                descriptors1 = descriptors1.cuda()
                kps2 = kps2.cuda()
                descriptors2 = descriptors2.cuda()

            # 使用 feature_booster 增强描述符
            out1 = feature_booster(descriptors1, kps1)
            out2 = feature_booster(descriptors2, kps2)
            if 'boost-b' in args.descriptor.lower():
                out1 = (out1 >= 0).cpu().detach().numpy()
                descriptors1 = np.packbits(out1, axis=1, bitorder='little')
            else:
                descriptors1 = out1.cpu().detach().numpy()
                descriptors2 = out2.cpu().detach().numpy()

        # save the features 打开一个新文件以二进制写入模式，文件名由图像路径和描述符参数组成
        with open(path1 + '.' + args.descriptor, 'wb') as output_file1:
            np.savez( # 使用NumPy的 np.savez 函数将特征点和描述符保存到文件中
                output_file1,
                keypoints=keypoints1,
                descriptors=descriptors1
            )
        with open(path2 + '.' + args.descriptor, 'wb') as output_file2:
            np.savez( # 使用NumPy的 np.savez 函数将特征点和描述符保存到文件中
                output_file2,
                keypoints=keypoints2,
                descriptors=descriptors2
            )
