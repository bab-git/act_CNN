# encoding: utf-8

"""
Loading the data
"""
import torch
from loader.loader import loader
import numpy as np

def fetch_dataloader(types, in_arg):
    """
    Fetch and return train/dev
    """
    if 'NTU-RGB-D' in in_arg.dataset_name :
        if 'CV' in in_arg.dataset_name:
            in_arg.train_loader_args["data_path"] = in_arg.dataset_dir+'NTU-RGB-D'+'/xview/train_data.npy'
            in_arg.train_loader_args["num_frame_path"] = in_arg.dataset_dir+'NTU-RGB-D'+'/xview/train_num_frame.npy'
            in_arg.train_loader_args["label_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xview/train_label.pkl'
            in_arg.test_loader_args["data_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xview/val_data.npy'
            in_arg.test_loader_args["num_frame_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xview/val_num_frame.npy'
            in_arg.test_loader_args["label_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xview/val_label.pkl'

        if 'CS' in in_arg.dataset_name:
            in_arg.train_loader_args["data_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xsub/train_data.npy'
            in_arg.train_loader_args["num_frame_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xsub/train_num_frame.npy'
            in_arg.train_loader_args["label_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xsub/train_label.pkl'
            in_arg.test_loader_args["data_path"]= in_arg.dataset_dir + 'NTU-RGB-D' + '/xsub/val_data.npy'
            in_arg.test_loader_args["num_frame_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xsub/val_num_frame.npy'
            in_arg.test_loader_args["label_path"] = in_arg.dataset_dir + 'NTU-RGB-D' + '/xsub/val_label.pkl'



    if types == 'train':
        if not hasattr(in_arg,'batch_size_train'):
            in_arg.batch_size_train = in_arg.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=loader(**in_arg.train_loader_args),
            batch_size=in_arg.batch_size_train,
            shuffle=True,
            num_workers=in_arg.num_workers,pin_memory=in_arg.cuda)

    if types == 'test':
        if not hasattr(in_arg,'batch_size_test'):
            in_arg.batch_size_test = in_arg.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=loader(**in_arg.test_loader_args),
            batch_size=in_arg.batch_size_test ,
            shuffle=False,
            num_workers=in_arg.num_workers,pin_memory=in_arg.cuda)

    return loader

if __name__ == '__main__':

    pass
