#!/usr/bin/env python3

# Script for automatically tuning the hyper-parameters.
# You can usually get better results if you tune your model *manually*, though.

from __future__ import division
from __future__ import print_function

import subprocess
import sys

import hyperopt

min_y = 0
min_c = None


def flush():
    sys.stdout.flush()
    sys.stderr.flush()


def trial(hyperpm):
    global min_y, min_c
    cmd = 'CUDA_VISIBLE_DEVICES=3 python main.py'
    cmd += ' --data data/alishop --epoch 50 --dfac 100'
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if isinstance(v, str):
            cmd += ' %s' % v
        elif int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        print('\nval=.....%% tst=.....%% @ %s' % cmd, file=sys.stderr)
        flush()
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        print('...')
        flush()
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    val, tst = eval(output)
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    flush()
    score = -val  # tune hyper-parameters according to validation performance
    if score < min_y:
        min_y, min_c = score, cmd
    return {'loss': score, 'status': hyperopt.STATUS_OK}


# You may need to modify the search space listed below before tuning your model.
space = {
    'lr': hyperopt.hp.loguniform('lr', -8, 0),
    'rg': hyperopt.hp.loguniform('rg', -12, 0),
    'keep': hyperopt.hp.quniform('dropout', 1, 20, 1) * 0.05,
    'beta': hyperopt.hp.quniform('beta', 1, 100, 1) * 0.05,
    'tau': hyperopt.hp.quniform('tau', 1, 20, 1) * 0.05,
    'std': hyperopt.hp.quniform('std', 1, 10, 1) * 0.025,
    'kfac': hyperopt.hp.quniform('kfac', 1, 10, 1)
}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
print('>>>>', min_c)
