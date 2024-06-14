import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch

# from dog import Dog
from featurebooster import FeatureBooster

import sys
from pathlib import Path

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
        help='descriptor to extract'
    )
    
    parser.add_argument(
        '--image_list_file', type=str, default=Path(__file__).parent / "change_detection_list.txt",
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
        model_path = Path(__file__).parent / str("models/SuperPoint+Boost-F.pth")
        # print(model_path)
        model_dict = torch.load(model_path)
        feature_booster.load_state_dict(model_dict)

    # Process the file 处理图像文件并提取特征
    with open(args.image_list_file, 'r') as f: # 打开一个包含图像路径列表的文件
        lines = f.readlines()

    for line in tqdm(lines, total=len(lines)): # 遍历每一行（每个图像路径）
        path = line.strip()
        image = cv2.imread(path)
        # file_name = os.path.basename(path)
        # new_path = os.path.join("D:/study/data/change/output/sppF", file_name)


        if 'alike' in args.descriptor.lower():
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred = feature_extractor(rgb, sub_pixel=True)
            keypoints = pred['keypoints']
            descriptors = pred['descriptors']
            scores = pred['scores']
            keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if 'superpoint' in args.descriptor.lower():
                image = (image.astype('float32') / 255.)
                keypoints, descriptors, _ = feature_extractor.run(image)
                keypoints, descriptors = keypoints.T, descriptors.T
            elif args.descriptor.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
                image = (image.astype('float32') / 255.)
                keypoints, scores, descriptors = feature_extractor.detectAndCompute(image)
                keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
            elif 'sift' in args.descriptor.lower():
                image = (image.astype('float32') / 255.)
                keypoints, scores, descriptors = feature_extractor.detectAndCompute(image)
            elif 'orb' in args.descriptor.lower():
                kps_tuples, descriptors = feature_extractor.detectAndCompute(image)
                # convert keypoints 
                keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
                keypoints = np.array(
                    [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints], 
                    dtype=np.float32
                )

        if "+Boost-" in args.descriptor:
            # boosted the descriptor using trained model
            # 标准化特征点坐标
            kps = normalize_keypoints(keypoints, image.shape)
            kps = torch.from_numpy(kps.astype(np.float32))
            if 'orb' in args.descriptor.lower():
                descriptors = np.unpackbits(descriptors, axis=1, bitorder='little')
                descriptors = descriptors * 2.0 - 1.0
            descriptors = torch.from_numpy(descriptors.astype(np.float32))
            if use_cuda:
                kps = kps.cuda()
                descriptors = descriptors.cuda()

            # 使用 feature_booster 增强描述符
            out = feature_booster(descriptors, kps)
            if 'boost-b' in args.descriptor.lower():
                out = (out >= 0).cpu().detach().numpy()
                descriptors = np.packbits(out, axis=1, bitorder='little')
            else:
                descriptors = out.cpu().detach().numpy()

        # save the features 打开一个新文件以二进制写入模式，文件名由图像路径和描述符参数组成
        with open(path + '.' + args.descriptor, 'wb') as output_file:
            np.savez( # 使用NumPy的 np.savez 函数将特征点和描述符保存到文件中
                output_file,
                keypoints=keypoints,
                descriptors=descriptors
            )
