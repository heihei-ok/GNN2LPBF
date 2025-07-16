import os
import pickle
from sklearn.model_selection import train_test_split


class space_data_loda(object):
    def __init__(self, data_dir):
        self.dir = data_dir

    def data_pre(self, test=False):
        data_files = [f for f in os.listdir(self.dir) if 'data' in f]
        # inx_files = [f for f in os.listdir(self.dir) if 'inx' in f]
        data = []
        # inx = []
        for i in range(len(data_files)):
            data_path = os.path.join(self.dir, data_files[i])
            # inx_path = os.path.join(self.dir, inx_files[i])
            with open(data_path, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
            # with open(inx_path, 'rb') as fo:
            #     list_inx = pickle.load(fo, encoding='bytes')
            data += list_data
            # inx += list_inx

        train_dataset, dataset = train_test_split(data, test_size=0.4, random_state=40)
        val_dataset, test_dataset = train_test_split(dataset, test_size=0.5, random_state=40)
        if test:
            return test_dataset
        else:
            return train_dataset, val_dataset


if __name__ == "__main__":
    x = space_data_loda().data_pre()
