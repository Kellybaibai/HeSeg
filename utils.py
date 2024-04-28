'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import random
import math
import numpy as np
import pandas as pd
import shutil
import pickle
import torch.nn as nn
import torch.nn.init as init
import torch
from contextlib import contextmanager

def label_encode(series):
    """
    for col in cat_cols:
        df_ads[col]=label_encode(df_ads[col])
    """
    unique = list(series.unique())
    return series.map(dict(zip(unique, range(0,series.nunique()))))

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None, pre_msg=''):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    current+=1
    
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    eta_time = step_time*(total-current)

    L = []
    L.append(' Time:%s' % format_time2(tot_time))
    L.append(' | ETA:%s' % format_time2(eta_time))
    if msg:
        L.append(' | ' + msg)

    msg_ = ''.join(L)

    columns, rows = os.get_terminal_size(0)
    TOTAL_BAR_LENGTH=columns-len(msg_)-len(pre_msg)-3
    if TOTAL_BAR_LENGTH<20:
        L = []
        L.append(' Time:%s' % format_time(tot_time))
        L.append(' | ETA:%s' % format_time(eta_time))
        if msg:
            L.append(' | ' + msg)

        msg_ = ''.join(L)

        msg_=msg_.replace(' ','')
        msg_=msg_.replace(':','')
        pre_msg=pre_msg.replace(' ','')
        pre_msg=pre_msg.replace(':','')
        
        TOTAL_BAR_LENGTH=columns-len(msg_)-len(pre_msg)-3
    TOTAL_BAR_LENGTH=np.clip(TOTAL_BAR_LENGTH,5,80)

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    sys.stdout.write(pre_msg)
    
    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    
    sys.stdout.write(msg_)
    
    space_len=columns-(len(pre_msg)+TOTAL_BAR_LENGTH+len(msg_)+2)-1
    for i in range(space_len):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    pct=' %d/%d ' % (current, total)
    if TOTAL_BAR_LENGTH>len(pct):
        for i in range(int(TOTAL_BAR_LENGTH/2+len(pct)/2)+len(msg_)+space_len):
            sys.stdout.write('\b')
        sys.stdout.write(pct)

    if current < total:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*10)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += "%2d" % hours + ':'
        i += 1
    if minutes > 0 and i <= 2:
        f += "%2d" % minutes + ':'
        i += 1
    if secondsf >= 0 and i <= 2:
        f += "%2d" % secondsf + '.'
        i += 1
    if millis >= 0 and i <= 2:
        f += "%01d" % millis
        i += 1
    if f == '':
        f = '0.0'
    return f


def format_time2(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    if days > 0:
        f += str(days) + 'D'
    if hours > 0:
        f += "%2d" % hours + ':'
    if minutes > 0:
        f += "%2d" % minutes + ':'
    if secondsf >= 0:
        f += "%2d" % secondsf + '.'
    f += "%03d" % millis
    return f


def to_pickle(x,path):
    with open(path, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

def from_pickle(path):
    with open(path, 'rb') as f:
        x=pickle.load(f, encoding='latin1')
    return x

@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f'[{name}] {elapsed: .3f}sec')

def RemoveDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)