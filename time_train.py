import argparse
import os
from utils.train_graph_utils import train_utils
from datetime import datetime

args = None
from utils.logger import setlogger
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='GCN',
                        choices=['GCN', 'HoGCN', 'ChebyNet', 'SGCN', 'GAT', 'GIN', 'GraphSage'],
                        help='the name of the model')
    parser.add_argument('--data_dir', type=str, default="F:\data\sec_data",
                        help='the directory of the source data_space')
    parser.add_argument('--dataset_dir', type=str, default="time_based/data_time",
                        help='the directory of the process dataset')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--num_classes', type=int, default=5, help='the number of classes')
    parser.add_argument('--feature', type=int, default=128, help='the node feature dim of represent')
    parser.add_argument('--path_size', type=int, default=[3, 3], help='the size of graph')

    # Define the tasks
    parser.add_argument('--graph_type', type=str, choices=['KNNGraph', 'RadiusGraph', 'PathGraph'],
                        default='KNNGraph', help='the type of the graph')
    parser.add_argument('--task', choices=['Space', 'Time'], type=str,
                        default='Time', help='Spatial-based graph classification or Time-based graph classification')
    parser.add_argument('--graph_task', choices=['Node', 'Graph'], type=str,
                        default='Node', help='ph classification')
    # optimization information
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='the weight decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='10', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # names = ['HoGCN', 'ChebyNet', 'SGCN', 'GAT', 'GIN', 'GraphSage']
    # for name in names:
    #     args.model_name = name
        # Prepare the saving path for the model
    sub_dir = args.task + '_' + args.model_name + '_'+ args.graph_type + '_'+ str(args.path_size[0]) + '-' + \
              str(args.path_size[1])  + datetime.strftime(
        datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # log_path = args.model_name + '_train.log'
    setlogger(os.path.join(save_dir, 'train.log'))
    # setlogger(os.path.join(save_dir, log_path))
    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
    trainer.test()
    trainer.tsne()

