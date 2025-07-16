import os
import pickle
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from utils.generator import Gen_graph
import time

class time_data_trans(object):
    def __init__(self, datapath, graph_type, path_size, garaph_task):
        self.graph_type = graph_type
        self.class_name = ['E1', 'E2', 'E3', 'E4', 'E5']
        self.datapath = datapath
        self.num_line = 126
        self.path_size = path_size
        self.task = garaph_task

    def data_path(self):
        """
        获取每类零件下每层文件数据的路径，每层数据文件按层数命名，无需再次排序
        :return:
        """
        pathlab = []
        sets = os.listdir(self.datapath)
        for se in sets:
            path0 = os.path.join(self.datapath, se) + '\ch1'
            path = os.listdir(path0)
            for inx in range(len(path)):
                path[inx] = os.path.join(path0, path[inx])
            pathlab.append(path)
        return pathlab

    def lay_cut(self, data, num_segments):
        segment_size = len(data) // num_segments
        segments = [data[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
        return segments, segment_size // 3

    def line_cut(self, data, num_segments):
        segment_size = len(data) // num_segments
        segments = [np.reshape(data[i * segment_size:(i + 1) * segment_size], -1) for i in range(num_segments)]
        return segments

    def data_down(self, data_air):
        paths = self.data_path()
        star = time.time()
        for clc_path in tqdm(range(len(paths)), desc="Processing", unit="item"):
            lay_paths = paths[clc_path]
            x = []
            for num_lay in range(len(lay_paths)):
                layer_data = loadmat(lay_paths[num_lay])[self.class_name[clc_path]]
                line_sques, dim_3 = self.lay_cut(layer_data, self.num_line)
                data_matix = []
                for inx, line_data in enumerate(line_sques):
                    lines = self.line_cut(line_data, 3)
                    data_matix.extend(lines)
                x.extend(data_matix)
            data_list = []
            a, b = 0, self.path_size[0] * self.path_size[1]
            while b <= len(x):
                data_list.append(x[a:b])
                a += self.path_size[0] * self.path_size[1]
                b += self.path_size[0] * self.path_size[1]
            graphset = Gen_graph(self.graph_type, data_list, clc_path, self.task)
            with open(os.path.join(data_air, 'data_' + self.graph_type + str(clc_path) + ".pkl"), 'wb') as fo:
                pickle.dump(graphset, fo)
        print("Cost time:", time.time()-star)


if __name__ == "__main__":
    a = time_data_trans("F:\data\sec_data")
    a.data_down('F:\paper\graph\mygraph_exit\data_time')
