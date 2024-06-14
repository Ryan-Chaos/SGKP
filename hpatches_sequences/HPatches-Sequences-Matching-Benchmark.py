import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.io import loadmat
# from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Add new methods here.
methods = ['SuperPoint+Boost-F']
names = ['SuperPoint+Boost-F']
# methods = ['SuperPoint']
# names = ['SuperPoint']
colors = ['red']
linestyles = ['--']

n_i = 847
n_v = 3

# dataset_path = 'hpatches-sequences-release'
dataset_path = 'D:/work/data/SECOND_val_set/outputs'
save_path = 'D:/work/data/SECOND_val_set/match_images3'

lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t() # 计算两组特征点的相似度（即描述符之间的相似性）
    nn12 = torch.max(sim, dim=1)[1] # 对于每个特征点，找到与其相似度最高的特征点
    nn21 = torch.max(sim, dim=0)[1] # nn12 和 nn21，它们分别表示了 descriptors_a 中每个特征点与 descriptors_b 中最相似的特征点的索引
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12]) # 创建一个掩码 mask，其中 mask[i] 为 True 表示第 i 个特征点在两组特征点中都有相似的最佳匹配
    matches = torch.stack([ids1[mask], nn12[mask]]) # 将匹配成功的特征点对的索引存储在 matches 中，并将其转换为 NumPy 数组返回


    return matches.t().data.cpu().numpy()

def draw_matches(image1, match_kp1, image2, match_kp2):
    """
    这个函数用于在一对图像上绘制匹配的关键点。它接受两个图像(image1 和 image2)和它们各自的匹配关键点(match_kp1 和 match_kp2)
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    out_image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8) # 创建一个足够大的空白图像(out_image)来容纳两个图像并排显示
    out_image[:h1, :w1, :] = image1
    out_image[:h2, w1:w1+w2, :] = image2

    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2) in zip(match_kp1, match_kp2): # 使用循环遍历每对匹配的关键点，并在它们的位置上绘制黄色圆圈(yellow)和绿色连线(green)
        cv2.circle(out_image, (int(x1), int(y1)), 6, yellow, -1)
        cv2.circle(out_image, (int(x2) + w1, int(y2)), 6, yellow, -1)

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_color = red if distance > 50 else green

        # cv2.line(out_image, (int(x1), int(y1)), (int(x2) + w1, int(y2)), green, 2)
        cv2.line(out_image, (int(x1), int(y1)), (int(x2) + w1, int(y2)), line_color, 2)
    return out_image

# 这里是评估的主要函数
def benchmark_features(read_feats):
    seq_names = sorted(os.listdir(dataset_path))  # ：这行代码列出了dataset_path目录中的所有文件和文件夹，并将它们排序

    # 初始化了几个列表和字典来存储特征点数量、匹配数量、序列类型以及不同阈值下的错误率和匹配数
    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    i_matches = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}
    v_matches = {thr: 0 for thr in rng}

    half_length = len(seq_names) // 2
    for seq_idx, seq_name in tqdm(enumerate(seq_names[:half_length]), total=half_length):
        keypoints_a, descriptors_a = read_feats(seq_name)  # 对于每个序列，读取第一幅图像的关键点和描述符

        content = seq_name.split("_")[-1].split(".SuperPoint")[0]

        r1 = 'D:/work/data/SECOND_val_set/im1/' + content
        r2 = 'D:/work/data/SECOND_val_set/im2/' + content
        n_feats.append(keypoints_a.shape[0])

        new_seq_name = seq_name.replace("1_", "2_")
        keypoints_b, descriptors_b = read_feats(new_seq_name)
        n_feats.append(keypoints_b.shape[0])

        matches = mnn_matcher(  # 使用最近邻匹配器找到描述符之间的匹配
            torch.from_numpy(descriptors_a).to(device=device),
            torch.from_numpy(descriptors_b).to(device=device)
        )

        # txtpath = os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx))
        # homography = np.loadtxt(txtpath)  # 加载两幅图像之间的单应性矩阵
        homography = np.eye(3)

        # 计算匹配点之间的位置误差
        pos_a = keypoints_a[matches[:, 0], : 2]  # 从关键点数组keypoints_a中提取出所有匹配的第一组关键点的位置信息
        pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])],
                                 axis=1)  # 这里将pos_a的每个点扩展为齐次坐标形式，即在每个点的x和y坐标后面添加了一个1
        pos_b_proj_h = np.transpose(
            np.dot(homography, np.transpose(pos_a_h)))  # 通过单应性矩阵homography将pos_a_h中的点投影到第二幅图像的坐标系中
        pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]  # 将投影点的齐次坐标转换回普通坐标

        pos_b = keypoints_b[matches[:, 1], : 2]  # 从关键点数组keypoints_b中提取出所有匹配的第二组关键点的位置信息

        image1 = cv2.imread(r1)
        image2 = cv2.imread(r2)
        matched_image = draw_matches(image1, pos_a, image2, pos_b)


        cv2.imwrite(os.path.join(save_path, content), matched_image)

        dist = np.sqrt(
            np.sum((pos_b - pos_b_proj) ** 2, axis=1))  # 计算每对匹配点之间的欧氏距离，这是通过取实际点pos_b和投影点pos_b_proj之间差的平方和的平方根来实现的

        n_matches.append(matches.shape[0])  # 将当前匹配的数量添加到n_matches列表中
        seq_type.append(seq_name[0])  # 将当前序列的类型（例如，如果序列名称以'i'开头，则为内部序列）添加到seq_type列表中

        if dist.shape[0] == 0:
            dist = np.array([float("inf")])

        for thr in rng:  # 计算不同阈值下的平均误差和匹配数
            i_err[thr] += np.mean(dist <= thr)
            i_matches[thr] += np.sum(dist <= thr)
            v_err[thr] += 0
            v_matches[thr] += 0
            # if seq_name[0] == 'i':
            #     i_err[thr] += np.mean(dist <= thr)
            #     i_matches[thr] += np.sum(dist <= thr)
            # else:
            #     v_err[thr] += np.mean(dist <= thr)
            #     v_matches[thr] += np.sum(dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    # v_err[thr] = 0
    # v_matches[thr] += 0

    return i_err, v_err, i_matches, v_matches, [seq_type, n_feats, n_matches]
    # return i_err,0, i_matches,0,[seq_type, n_feats, n_matches]

def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}'.format(
        np.sum(n_matches) / ((n_i)),
        np.sum(n_matches[seq_type == 'i']) / (n_i))
    )

def getBit(des):
    res = []
    for d in des:
        for i in range(8):
            res.append(((d >> i) & 1) * 2 - 1)
    return res

def generate_read_function(method, extension='ppm', type='float'): # 创建读取图像数据的函数
    def read_function(seq_name):
        # path = os.path.join(dataset_path, seq_name, '%d.%s.%s' % (im_idx, extension, method))
        path = os.path.join(dataset_path, seq_name)
        aux = np.load(path) # 从文件中加载数据
        if type == 'float': # 如果参数 type 为 'float'，则函数返回从加载的数据中得到的 'keypoints' 和 'descriptors'
            return aux['keypoints'], aux['descriptors']
        else: # 处理 'descriptors' 数据，通过解包其位并缩放值
            descriptors = np.unpackbits(aux['descriptors'], axis=1, bitorder='little')
            descriptors = descriptors * 2.0 - 1.0
            return aux['keypoints'], descriptors
    return read_function

def sift_to_rootsift(descriptors):
    return np.sqrt(descriptors / np.expand_dims(np.sum(np.abs(descriptors), axis=1), axis=1) + 1e-16)
def parse_mat(mat):
    keypoints = mat['keypoints'][:, : 2]
    raw_descriptors = mat['descriptors']
    l2_norm_descriptors = raw_descriptors / np.expand_dims(np.sum(raw_descriptors ** 2, axis=1), axis=1)
    descriptors = sift_to_rootsift(l2_norm_descriptors)
    return keypoints, descriptors

cache_dir = 'cache'
if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)

errors = {}
for method in methods:
    output_file = os.path.join(cache_dir, method + '.npy')
    print(method)
    if method == 'hesaff':
        read_function = lambda seq_name, im_idx: parse_mat(loadmat(os.path.join(dataset_path, seq_name, '%d.ppm.hesaff' % im_idx), appendmat=False))
    else:
        if method == 'delf' or method == 'delf-new':
            read_function = generate_read_function(method, extension='png')
        elif '+Boost-B' in method or (method.lower() == 'orb'):
            read_function = generate_read_function(method, type='binary')
        else:
            read_function = generate_read_function(method)  # 创建读取图像数据的函数read_function
    # if os.path.exists(output_file):
    #     print('Loading precomputed errors...')
    #     errors[method] = np.load(output_file, allow_pickle=True)# 如果文件存在，使用 np.load 加载文件数据到 errors字典中xxxx
    # else: # 主要在这里
    errors[method] = benchmark_features(read_function) # 如果不存在，调用 benchmark_features 函数对特征进行评估，并将结果存储在 errors 字典中
    np.save(output_file, errors[method])
    summary(errors[method][-1]) # 调用 summary 函数，传入 errors 字典中当前方法的最后一个元素，以生成和打印性能摘要

for method in methods:
    i_err, v_err, i_matches, v_matches, _ = errors[method]
    # 从 errors 字典中提取当前方法的错误和匹配信息，i_err 和 v_err 分别代表在不同阈值下的照明和视角错误，i_matches 和 v_matches 代表匹配的内点数
    print(method)
    for thr in [1, 3, 5]:
        print('# MMA@{:d}: Overall {:f}'.format( # 这里计算了整体的MMA、照明条件下的MMA和视角变化下的MMA
            thr,
            # (i_err[thr] + v_err[thr]) / n_features)
            (i_err[thr]/n_i)*n_v)
        )
        print('# inliers@{:d}: Overall {:f}'.format( # 这里计算了整体的内点数、照明条件下的内点数和视角变化下的内点数
            thr,
            (i_matches[thr]/n_i)*n_v)
        )

