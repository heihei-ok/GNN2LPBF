import os
import pickle
import numpy as np
from scipy.io import loadmat
from utils.generator import data_cut
from tqdm import tqdm
import time
# Reconstructing the code for data loading and data splitting
# based on your data storage format
class space_data_trans(object):
    def __init__(self, datapath, path_size, num_connect, task):
        self.class_name = ['You', 'class', 'Name']
        self.datapath = datapath
        self.num_line = 126
        self.path_size = path_size
        self.num_connect = num_connect
        self.task = task

    def data_path(self):
        pathlab = []
        sets = os.listdir(self.datapath)
        for se in sets:
            path0 = os.path.join(self.datapath, se)
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
        segments = [data[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
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
                data_matix = np.ones([126, 3, dim_3])
                for inx, line_data in enumerate(line_sques):
                    inx_xy = np.zeros([126, 3])
                    lines = self.line_cut(line_data, 3)
                    if inx % 2 == 0:
                        data_matix[inx, 0, :] = lines[0].reshape(-1)
                        data_matix[inx, 1, :] = lines[1].reshape(-1)
                        data_matix[inx, 2, :] = lines[2].reshape(-1)
                        inx_xy[inx, 0] = 0
                        inx_xy[inx, 1] = 1
                        inx_xy[inx, 2] = 2
                    else:
                        data_matix[inx, 0, :] = lines[2].reshape(-1)
                        data_matix[inx, 1, :] = lines[1].reshape(-1)
                        data_matix[inx, 2, :] = lines[0].reshape(-1)
                        inx_xy[inx, 0] = 2
                        inx_xy[inx, 1] = 1
                        inx_xy[inx, 2] = 0
                x1 = data_cut(data_matix, self.path_size, self.num_connect, clc_path, self.task)
                x = x + x1
            with open(os.path.join(data_air, 'data_' + str(clc_path) + ".pkl"), 'wb') as fo:
                pickle.dump(x, fo)
        print("Cost time:", time.time()-star)
