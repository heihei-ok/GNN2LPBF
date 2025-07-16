import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import nets
import models
import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch_geometric.data import DataLoader
from space_based.space_dataload.space_data_loda import space_data_loda
from space_based.space_dataload.datatrans import space_data_trans
from time_based.time_dataload.time_data_loda import time_data_loda
from time_based.time_dataload.datatrans import time_data_trans

from utils.callbacks import LossHistory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class train_utils(object):
    def __init__(self, args, save_dir):
        self.datasets = None
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args
        self.start_epoch = 0
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        self.datasets = {}
        # 判断图数据是否生成

        # 数据载入
        if args.task == 'Space':
            # data_file = any(os.path.isfile(os.path.join(args.dataset_dir, f)) for f in os.listdir(args.dataset_dir))
            datatrans = space_data_trans(args.data_dir, args.path_size, args.path_connect, args.graph_task)
            datatrans.data_down(args.dataset_dir)
            self.datasets['train'], self.datasets['val'] = space_data_loda(args.dataset_dir).data_pre()
        else:
            datatrans = time_data_trans(args.data_dir, args.graph_type, args.path_size, args.graph_task)
            datatrans.data_down(args.dataset_dir)
            self.datasets['train'], self.datasets['val'] = time_data_loda(args.dataset_dir).data_pre(args.graph_type)

        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        if args.graph_task == 'Node':
            self.model = getattr(models, args.model_name)(feature=args.feature, out_channel=args.num_classes)
        elif args.graph_task == 'Graph':
            self.model = getattr(nets, args.model_name)(feature=args.feature, out_channel=args.num_classes)
        else:
            print('The task is wrong!')
        # Define the optimizer
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                   momentum=args.momentum, weight_decay=args.weight_decay)
        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
        #                             weight_decay=args.weight_decay)
        # # Define the learning rate decay
        # steps = int(args.steps)
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        steps = int(args.steps)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, gamma=args.gamma)
        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """
        Training process
        :return:
        """
        self.callback = LossHistory(self.save_dir)
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                sample_num = 0
                for data in self.dataloaders[phase]:
                    inputs = data.to(self.device)
                    labels = inputs.y
                    if args.graph_task == 'Node':
                        bacth_num = inputs.num_nodes
                        sample_num += len(labels)
                    elif args.graph_task == 'Graph':
                        bacth_num = inputs.num_graphs
                        sample_num += len(labels)
                    else:
                        print("There is no such task!!")
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * bacth_num
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += bacth_num
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {}, Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_loss, batch_acc, sample_per_sec, batch_time
                                ))

                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / sample_num
                epoch_acc = epoch_acc / sample_num

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))
                self.callback.append_data(epoch_loss, epoch_acc, phase)
                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
                        self.best_path = os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc))

                    if epoch > args.max_epoch - 2:
                        logging.info("save last model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-last_model.pth'.format(epoch, epoch_acc)))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def test(self, model_path=None):
        args = self.args
        if model_path is None:
            model_path = self.best_path
        if args.graph_task == 'Node':
            model = getattr(models, args.model_name)(feature=args.feature, out_channel=args.num_classes).cuda()
        elif args.graph_task == 'Graph':
            model = getattr(nets, args.model_name)(feature=args.feature, out_channel=args.num_classes).cuda()
        else:
            print('The task is wrong!')
        model_dict = torch.load(model_path)
        logging.info('Note: The model path  “ {} ” has be loaded! '.format(model_path))
        matched_state_dict = {k: v for k, v in model_dict.items() if k in model.state_dict()}
        model.load_state_dict(matched_state_dict)
        model.eval()
        datasets = {}
        data_file = any(os.path.isfile(os.path.join(args.dataset_dir, f)) for f in os.listdir(args.dataset_dir))
        if args.task == 'Space':
            if not data_file:
                datatrans = space_data_trans(args.data_dir)
                datatrans.data_down(args.dataset_dir)
                datasets['test'] = space_data_loda(args.dataset_dir).data_pre(test=True)
            else:
                datasets['test'] = space_data_loda(args.dataset_dir).data_pre(test=True)
        else:
            if not data_file:
                datatrans = time_data_trans(args.data_dir, args.graph_type)
                datatrans.data_down(args.dataset_dir)
                datasets['test'] = time_data_loda(args.dataset_dir).data_pre(args.graph_type, test=True)
            else:
                datasets['test'] = time_data_loda(args.dataset_dir).data_pre(args.graph_type, test=True)
        dataloaders = DataLoader(datasets['test'], batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 pin_memory=(True if self.device == 'cuda' else False))
        step_start = time.time()
        preds = []
        labels = []
        for data in dataloaders:
            inputs = data.to(self.device)
            label = inputs.y
            labels.extend(label.detach().cpu().numpy())
            with torch.set_grad_enabled(False):
                # forward
                logits = model(inputs)
                pred = logits.argmax(dim=1)
                preds.extend(pred.detach().cpu().numpy())
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        f1 = f1_score(labels, preds, average="macro")
        logging.info('The test Cost {:.4f} sec'.format(
            time.time() - step_start))
        logging.info('Test Result: Accuracy: {:.4f}, Precision: {:.4f}, Recall {:.4f}, F1 {:.4f}'.format(
            accuracy, precision, recall, f1))

    def tsne(self, model_path=None):
        args = self.args
        if model_path is None:
            model_path = self.best_path
        if args.graph_task == 'Node':
            model = getattr(models, args.model_name)(feature=args.feature, out_channel=args.num_classes).cuda()
        elif args.graph_task == 'Graph':
            model = getattr(nets, args.model_name)(feature=args.feature, out_channel=args.num_classes).cuda()
        else:
            print('The task is wrong!')
        model_dict = torch.load(model_path)
        logging.info('Note: The model path  “ {} ” has be loaded! '.format(model_path))
        matched_state_dict = {k: v for k, v in model_dict.items() if k in model.state_dict()}
        model.load_state_dict(matched_state_dict)
        model.eval()
        datasets = {}
        data_file = any(os.path.isfile(os.path.join(args.dataset_dir, f)) for f in os.listdir(args.dataset_dir))
        if args.task == 'Space':
            if not data_file:
                datatrans = space_data_trans(args.data_dir)
                datatrans.data_down(args.dataset_dir)
                datasets['test'] = space_data_loda(args.dataset_dir).data_pre(test=True)
            else:
                dataset = space_data_loda(args.dataset_dir).data_pre(test=True)
                datasetx, datasets['test'] = train_test_split(dataset, test_size=0.1, random_state=40)
        else:
            if not data_file:
                datatrans = time_data_trans(args.data_dir, args.graph_type)
                datatrans.data_down(args.dataset_dir)
                datasets['test'] = time_data_loda(args.dataset_dir).data_pre(args.graph_type, test=True)
            else:
                dataset = space_data_loda(args.dataset_dir).data_pre(test=True)
                datasetx, datasets['test'] = train_test_split(dataset, test_size=0.1, random_state=40)

        dataloaders = DataLoader(datasets['test'], batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 pin_memory=(True if self.device == 'cuda' else False))
        postconvolution = []
        labels = []

        # 定义钩子函数来捕获 fc2 层的输出
        def hook_fn(module, input, output):
            postconvolution.append(output.detach().cpu().numpy())

        hook = model.GConv2.register_forward_hook(hook_fn)

        for data in dataloaders:
            inputs = data.to(self.device)
            labels.extend(inputs.y.tolist())
            with torch.set_grad_enabled(False):
                # forward
                logits = model(inputs)
                pred = logits.argmax(dim=1)
        hook.remove()
        # 将捕获的输出转换为 numpy 数组并进行 t-SNE 降维
        if len(postconvolution[0].shape) == 3:
            fc2_outputs = np.concatenate(postconvolution, axis=0).squeeze(2)
        else:
            fc2_outputs = np.concatenate(postconvolution, axis=0)
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        reduced_features = tsne.fit_transform(fc2_outputs)
        # pca = PCA(n_components=2)
        # # 拟合数据并进行转换
        # reduced_features = pca.fit_transform(fc2_outputs)
        data = pd.DataFrame(reduced_features, columns=['Feature1', 'Feature2'])
        data['label'] = labels
        # 可视化 t-SNE 结果
        import matplotlib
        matplotlib.use('TkAgg')
        # 设置不同的形状和颜色
        shapes = ['o', 's', 'D', '^', 'P']
        colors = ['r', 'g', 'b', 'y', 'c']
        classes = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']
        plt.figure(figsize=(10, 8))
        for i in range(5):
            cluster_data = data[data['label'] == i]
            plt.scatter(cluster_data['Feature1'], cluster_data['Feature2'],
                        c=colors[i], marker=shapes[i], label=classes[i])
        plt.legend()
        plt.show()
        # saveself.save_dir
        # plt.savefig('sine_wave.png')
