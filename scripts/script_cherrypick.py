#!/usr/bin/env python3

# Script for cherry-picking an interpretable model.
#
# Cherry-picking is necessary if your model is an unsupervised one.
# I strongly recommend you to develop a (semi-)supervised model instead.
# Don't be as silly as me. Don't go the unsupervised route, so that
# you don't fall into this cherry-picking hell.


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import subprocess

import numpy as np

interpretable = 0  # the higher, the more interpretable

# Macro disentanglement is good when interpretable >=2,
# while good micro disentanglement requires a higher score.

while interpretable < 2:
    seed = np.random.randint(0, 1 << 30)
    cmd = 'CUDA_VISIBLE_DEVICES=3 python3 main.py '
    cmd += '--data data/alishop --epoch 20 --seed %d --mode %s'
    try:
        subprocess.check_output(cmd % (seed, 'trn'), shell=True)
        val = subprocess.check_output(cmd % (seed, 'vis'), shell=True)
        val = eval(val)
        print('interpretable=%d, seed=%d' % (val, seed))
    except subprocess.CalledProcessError:
        val = 0
    interpretable = max(interpretable, val)
