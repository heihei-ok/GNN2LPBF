import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data
from scipy.spatial.distance import pdist
import copy
from scipy.fftpack import dct


def slice_sequence(seq_a, seq_b):
    result = []
    start = 0
    for i in range(1, len(seq_a)):
      if seq_a[i] != seq_a[i-1]:
        result.append(seq_b[start:i])
        start = i
    result.append(seq_b[start:])
    return result

def get_patch_indices(input_shape, patch_size):
    H, W = input_shape
    ph, pw = patch_size

    # 计算每个 patch 的行列数
    num_patches_h = H // ph
    num_patches_w = W // pw

    # 存储每个 patch 的索引
    patch_indices = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # 每个 patch 的左上角位置
            start_h = i * ph
            start_w = j * pw

            # 每个 patch 中每个像素的索引
            for dh in range(ph):
                for dw in range(pw):
                    orig_h = start_h + dh
                    orig_w = start_w + dw
                    patch_indices.append((orig_h, orig_w))

    return np.array(patch_indices)


def edge_conect(x, y, num_connect):
    four_neighbors = {}
    eight_neighbors = {}
    inx4squ = {}
    inx8squ = {}
    edge_raw0 = []
    edge_raw1 = []
    # 遍历矩阵中的每个元素
    squ = 0
    for i in range(x):
        for j in range(y):
            # 初始化邻居列表
            four_connected = []
            # 上邻居
            if i > 0:
                four_connected.append((i - 1, j))
            # 下邻居
            if i < x - 1:
                four_connected.append((i + 1, j))
            # 左邻居
            if j > 0:
                four_connected.append((i, j - 1))
            # 右邻居
            if j < y - 1:
                four_connected.append((i, j + 1))
            # 将当前元素及其四连通邻居存储在字典中
            four_neighbors[(i, j)] = four_connected
            inx4squ[(i, j)] = squ

            eight_connected = four_connected[:]  # 先复制四连通邻居
            if i > 0 and j > 0:  # 左上
                eight_connected.append((i - 1, j - 1))
            if i > 0 and j < y - 1:  # 右上
                eight_connected.append((i - 1, j + 1))
            if i < x - 1 and j > 0:  # 左下
                eight_connected.append((i + 1, j - 1))
            if i < x - 1 and j < y - 1:  # 右下
                eight_connected.append((i + 1, j + 1))
            eight_neighbors[(i, j)] = eight_connected
            inx8squ[(i, j)] = squ
            squ += 1
    if num_connect == 4:
        for element, neighbors in four_neighbors.items():
            for link in neighbors:
                edge_raw0.append(inx4squ[element])
                edge_raw1.append(inx4squ[link])
    else:
        for element, neighbors in eight_neighbors.items():
            for link in neighbors:
                edge_raw0.append(inx8squ[element])
                edge_raw1.append(inx8squ[link])
    return [edge_raw0, edge_raw1]


def data_cut(data, path_size, num_connect, label, task):
    h, w, l = data.shape
    lay_data = []
    # 按照分区获取patch索引
    indic = get_patch_indices([h, w], path_size)
    edge = edge_conect(path_size[0], path_size[1], num_connect)
    # edge = edge_conect(3, 3, num_connect)
    len_cut = path_size[0] * path_size[1]
    for i in range(0, indic.shape[0], len_cut):
        patch_data = []
        w = []
        for j in range(len_cut):
            dct_image = dct(data[indic[i+j, 0], indic[i+j, 1], :], axis=0, norm='ortho')
            if len(dct_image) >= 128:
                ti = dct_image[:128]
            else:
                # 如果 DCT 系数少于所需数量，进行零填充
                ti = np.pad(dct_image, (0, 128 - len(dct_image)), 'constant')
            patch_data.append(ti)

        sliced_seq = slice_sequence(edge[0], edge[1])
        ac = 0
        for sqs in sliced_seq:
            distance = []
            for sq in range(len(sqs)):
                v_1 = patch_data[int(edge[0][ac + sq])]
                v_2 = patch_data[sqs[sq]]
                combine = np.vstack([v_1, v_2])
                likely = pdist(combine, 'euclidean')  # 计算欧几里得距离
                distance.append(likely[0])
            ac = ac+len(sqs)
            beata = np.mean(distance)
            m = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  # 使用高斯函数来计算权值
            w.extend(m)
        node_features = torch.tensor(patch_data, dtype=torch.float)
        if task == 'Node':
            labels = np.zeros(len(node_features)) + label
        elif task == 'Graph':
            labels = [label]
        graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
        edge_index = torch.tensor(edge, dtype=torch.long)
        edge_features = torch.tensor(w, dtype=torch.float)
        graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)  # 创建一个图对象
        lay_data.append(graph)
    return lay_data


def KNN_classify(k, X_set, x):
    """
    k:number of neighbours
    X_set: the datset of x
    x: to find the nearest neighbor of data_space x
    """
    # 欧几里得距离
    distances = [sqrt(np.sum((x_compare - x) ** 2)) for x_compare in X_set]
    nearest = np.argsort(distances)  # np.argsort返回了一个索引数组，这些索引能将 distances 数组按升序排列
    node_index = [i for i in nearest[1:k + 1]]  # 取出前k个索引，但不包括第一个，通常是自己
    topK_x = [X_set[i] for i in nearest[1:k + 1]]
    return node_index, topK_x


def KNN_weigt(x, topK_x):
    distance = []
    v_1 = x
    data_2 = topK_x
    for i in range(len(data_2)):
        v_2 = data_2[i]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')  # 计算欧几里得距离
        distance.append(likely[0])
    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  # 使用高斯函数来计算权值
    return w


def KNN_attr(data):
    '''
    for KNNgraph
    :param data:
    :return:
    '''
    edge_raw0 = []  # 单个图中 row0存储源节点，row1存储目标节点，fea存储节点链接边的权值
    edge_raw1 = []
    edge_fea = []
    for i in range(len(data)):
        x = data[i]
        node_index, topK_x = KNN_classify(5, data, x)  # 找到最近的K个节点
        loal_weigt = KNN_weigt(x, topK_x)  # 求解K个节点之间边的权值
        local_index = np.zeros(5) + i
        edge_raw0 = np.hstack((edge_raw0, local_index))
        edge_raw1 = np.hstack((edge_raw1, node_index))
        edge_fea = np.hstack((edge_fea, loal_weigt))

    edge_index = [edge_raw0, edge_raw1]

    return edge_index, edge_fea


def cal_sim(data, s1, s2):
    edge_index = [[], []]
    edge_feature = []
    if s1 != s2:
        v_1 = data[s1]
        v_2 = data[s2]
        combine = np.vstack([v_1, v_2])
        likely = 1 - pdist(combine, 'cosine')
        #         w = np.exp((-(likely[0]) ** 2) / 30)
        if likely.item() >= 0:
            w = 1
            edge_index[0].append(s1)
            edge_index[1].append(s2)
            edge_feature.append(w)
    return edge_index, edge_feature


def Radius_attr(data):
    '''
    for RadiusGraph
    :param feature:
    :return:
    '''
    s1 = range(len(data))  # [0，10)
    s2 = copy.deepcopy(s1)
    edge_index = np.array([[], []])  # 一个故障样本与其他故障样本匹配生成一次图
    edge_fe = []
    for i in s1:
        for j in s2:
            local_edge, w = cal_sim(data, i, j)
            edge_index = np.hstack((edge_index, local_edge))
            if any(w):
                edge_fe.append(w[0])
    return edge_index, edge_fe


def Path_attr(data):
    node_edge = [[], []]
    # 生成节点对：
    for i in range(len(data) - 1):
        node_edge[0].append(i)
        node_edge[1].append(i + 1)

    distance = []
    for j in range(len(data) - 1):
        v_1 = data[j]
        v_2 = data[j + 1]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])
    # 计算高斯核权重：
    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  # Gussion kernel高斯核

    return node_edge, w


def Gen_graph(graphType, data, label, task):
    data_list = []
    if graphType == 'KNNGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")

            #  生成ti图像
            for gi, line in enumerate(graph_feature):
                dct_image = dct(line, axis=0, norm='ortho')
                if len(dct_image) >= 128:
                    ti = dct_image[:128]
                else:
                    # 如果 DCT 系数少于所需数量，进行零填充
                    ti = np.pad(dct_image, (0, 128 - len(dct_image)), 'constant')
                # ti = get_ti(line, tao, len_win)
                graph_feature[gi] = ti
            node_edge, w = KNN_attr(graph_feature)
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)  # 创建一个图对象
            data_list.append(graph)

    elif graphType == 'RadiusGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")

            #  生成ti图像
            for gi, line in enumerate(graph_feature):
                dct_image = dct(line, axis=0, norm='ortho')
                if len(dct_image) >= 128:
                    ti = dct_image[:128]
                else:
                    # 如果 DCT 系数少于所需数量，进行零填充
                    ti = np.pad(dct_image, (0, 128 - len(dct_image)), 'constant')
                # ti = get_ti(line, tao, len_win)
                graph_feature[gi] = ti
            node_edge, w = Radius_attr(graph_feature)  # 10，1024 ——> graph
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    elif graphType == 'PathGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")

            #  生成ti图像
            #  生成ti图像
            for gi, line in enumerate(graph_feature):
                dct_image = dct(line, axis=0, norm='ortho')
                if len(dct_image) >= 128:
                    ti = dct_image[:128]
                else:
                    # 如果 DCT 系数少于所需数量，进行零填充
                    ti = np.pad(dct_image, (0, 128 - len(dct_image)), 'constant')
                # ti = get_ti(line, tao, len_win)
                graph_feature[gi] = ti
            node_edge, w = Path_attr(graph_feature)
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    else:
        print("This GraphType is not included!")
    return data_list
