import os
import shutil
import tarfile
import urllib.request

import h5py
import numpy as np
import scipy.io

DATA_URL = ('http://deepsea.princeton.edu/media/code/'
            'deepsea_train_bundle.v0.9.tar.gz')
DIR_PATH = './deepsea_train'
NUM_TRAIN_SETS = 10


def download():
    print(f'Downloading deepsea train bundle from {DATA_URL} ...')

    if not os.path.exists(DIR_PATH + '.tar.gz'):
        urllib.request.urlretrieve(DATA_URL, DIR_PATH + '.tar.gz')


def unzip():
    print(f'Unzipping deepsea train bundle into {DIR_PATH} ...')

    with tarfile.open(DIR_PATH + '.tar.gz') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f)


def move():
    print(f'Moving deepsea train/test/valid sets out of {DIR_PATH} ...')

    shutil.move(DIR_PATH + '/train.mat', './train.mat')
    shutil.move(DIR_PATH + '/test.mat', './test.mat')
    shutil.move(DIR_PATH + '/valid.mat', './valid.mat')


def mat2npy():
    print('Converting `.mat` to `.npy` data format ...')

    train_mat = h5py.File('./train.mat', 'r')
    np.save('./X_train', np.array(train_mat['trainxdata']).transpose(2, 1, 0))
    np.save('./y_train', np.array(train_mat['traindata']).T)
    print('Saved train')

    test_mat = scipy.io.loadmat('./test.mat')
    np.save('./X_test', np.array(test_mat['testxdata']))
    np.save('./y_test', np.array(test_mat['testdata']))
    print('Saved test')

    valid_mat = scipy.io.loadmat('./valid.mat')
    np.save('./X_valid', np.array(valid_mat['validxdata']))
    np.save('./y_valid', np.array(valid_mat['validdata']))
    print('Saved valid')


def split_train_set():
    print(f'Split train set into {NUM_TRAIN_SETS} distinct sets ...')

    os.mkdir('./train_sets')

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')

    X_train_len = X_train.shape[0]
    subset_len = X_train_len / NUM_TRAIN_SETS

    for i in range(NUM_TRAIN_SETS):
        begin = int(i * subset_len)
        end = int(begin + subset_len)

        X_train_i = X_train[begin:end]
        y_train_i = y_train[begin:end]

        np.save(f'./train_sets/X_train_set.{i + 1}', X_train_i)
        np.save(f'./train_sets/y_train_set.{i + 1}', y_train_i)
        print(f'Saved set {i + 1}')


def cleanup():
    print('Cleaning up and finishing ...')

    shutil.rmtree(DIR_PATH)
    os.remove('./train.mat')
    os.remove('./test.mat')
    os.remove('./valid.mat')


if __name__ == '__main__':
    download()
    unzip()
    move()
    mat2npy()
    split_train_set()
    cleanup()

    print('=' * 30)
    print('All done.')
