#!/usr/bin/env python3

# Code developed based on https://github.com/dawenl/vae_cf/

#  We also have a distributed version that uses sampled softmax.
#  However, that version requires Alibaba's internal infrastructure.
#  I'll see if I could manage to release the core part of that version
#  while removing the Alibaba-specific parts.

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import os
import shutil
import sys
import time

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing
import tensorflow as tf
from scipy import sparse
import tensorflow_probability as tfp
#from tensorflow.contrib.distributions import RelaxedOneHotCategorical
#from tensorflow.contrib.layers import apply_regularization, l2_regularizer
#import tf_slim as slim
ARG = argparse.ArgumentParser()
ARG.add_argument('--data', type=str, required=True,
                 help='./data/ml-latest-small, ./data/ml-1m, '
                      './data/ml-20m, or ./data/alishop-7c')
ARG.add_argument('--mode', type=str, default='trn',
                 help='trn/tst/vis, for training/testing/visualizing.')
ARG.add_argument('--logdir', type=str, default='./runs/')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed. Ignored if < 0.')
ARG.add_argument('--epoch', type=int, default=200,
                 help='Number of training epochs.')
ARG.add_argument('--batch', type=int, default=350,
                 help='Training batch size.')
ARG.add_argument('--lr', type=float, default=1e-3,
                 help='Initial learning rate.')
ARG.add_argument('--rg', type=float, default=0.0,
                 help='L2 regularization.')
ARG.add_argument('--keep', type=float, default=0.5,
                 help='Keep probability for dropout, in (0,1].')
ARG.add_argument('--beta', type=float, default=0.2,
                 help='Strength of disentanglement, in (0,oo).')
ARG.add_argument('--tau', type=float, default=0.1,
                 help='Temperature of sigmoid/softmax, in (0,oo).')
ARG.add_argument('--std', type=float, default=0.075,
                 help='Standard deviation of the Gaussian prior.')
#ARG.add_argument('--kfac', type=int, default=7,
#                 help='Number of facets (macro concepts).')
ARG.add_argument('--kfac', type=int, default=100,
                 help='Number of facets (macro concepts).')
ARG.add_argument('--dfac', type=int, default=100,
                 help='Dimension of each facet.')
ARG.add_argument('--nogb', action='store_true', default=False,
                 help='Disable Gumbel-Softmax sampling.')
ARG = ARG.parse_args()

if ARG.seed < 0:
    ARG.seed = int(time.time())
LOG_DIR = '%s-%dT-%dB-%glr-%grg-%gkp-%gb-%gt-%gs-%dk-%dd-%d' % (
    ARG.data.replace('-', '/'), ARG.epoch, ARG.batch, ARG.lr, ARG.rg, ARG.keep,
    ARG.beta, ARG.tau, ARG.std, ARG.kfac, ARG.dfac, ARG.seed)
if ARG.nogb:
    LOG_DIR += '-nogb'
LOG_DIR = os.path.join(ARG.logdir, LOG_DIR)

batch_size_vad = batch_size_test = ARG.batch


def set_rng_seed(seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


class MyVAE(object):
    def __init__(self, num_items):
        kfac, dfac = ARG.kfac, ARG.dfac
        self.lam = ARG.rg
        self.lr = ARG.lr
        self.random_seed = ARG.seed

        self.n_items = num_items

        # The first fc layer of the encoder Q is the context embedding table.
        self.q_dims = [num_items, dfac, dfac]
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(
                zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2  # mu & var
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            self.weights_q.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed)))
            bias_key = "bias_q_{}".format(i + 1)
            self.biases_q.append(tf.compat.v1.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

        self.items = tf.compat.v1.get_variable(
            name="items", shape=[num_items, dfac],
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed))

        self.cores = tf.compat.v1.get_variable(
            name="cores", shape=[kfac, dfac],
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed))

        self.input_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, num_items])
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1., shape=None)
        self.is_training_ph = tf.compat.v1.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.compat.v1.placeholder_with_default(1., shape=None)

    def build_graph(self, save_emb=False):
        if save_emb:
            saver, facets_list = self.forward_pass(save_emb=True)
            return saver, facets_list, self.items, self.cores

        saver, logits, recon_loss, kl = self.forward_pass(save_emb=False)

        #reg_var = apply_regularization(l2_regularizer(self.lam), self.weights_q + [self.items, self.cores])
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_elbo = recon_loss + self.anneal_ph * kl # + 2. * reg_var

        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(neg_elbo)

        # add summary statistics
        tf.compat.v1.summary.scalar('trn/neg_ll', recon_loss)
        tf.compat.v1.summary.scalar('trn/kl_div', kl)
        tf.compat.v1.summary.scalar('trn/neg_elbo', neg_elbo)
        merged = tf.compat.v1.summary.merge_all()

        return saver, logits, train_op, merged

    def q_graph_k(self, x):
        mu_q, std_q, kl = None, None, None
        h = tf.nn.l2_normalize(x, 1)
        h = tf.nn.dropout(h, 1 - (self.keep_prob_ph))
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w, a_is_sparse=(i == 0)) + b
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                mu_q = tf.nn.l2_normalize(mu_q, axis=1)
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = ARG.std
                std_q = tf.exp(0.5 * lnvarq_sub_lnvar0) * std0
                # Trick: KL is constant w.r.t. to mu_q after we normalize mu_q.
                kl = tf.reduce_mean(input_tensor=tf.reduce_sum(
                    input_tensor=0.5 * (-lnvarq_sub_lnvar0 + tf.exp(lnvarq_sub_lnvar0) - 1.),
                    axis=1))
        return mu_q, std_q, kl

    def forward_pass(self, save_emb):
        # clustering
        print("clustering...1234")
        cores = tf.nn.l2_normalize(self.cores, axis=1)
        items = tf.nn.l2_normalize(self.items, axis=1)
        cates_logits = tf.matmul(items, cores, transpose_b=True) / ARG.tau
        if ARG.nogb:
            cates = tf.nn.softmax(cates_logits, axis=1)
        else:
            cates_dist = tfp.distributions.RelaxedOneHotCategorical(1, cates_logits)
            cates_sample = cates_dist.sample()
            cates_mode = tf.nn.softmax(cates_logits, axis=1)
            cates = (self.is_training_ph * cates_sample +
                     (1 - self.is_training_ph) * cates_mode)

        z_list = []
        probs, kl = None, None
        for k in range(ARG.kfac):
            cates_k = tf.reshape(cates[:, k], (1, -1))

            # q-network
            x_k = self.input_ph * cates_k
            mu_k, std_k, kl_k = self.q_graph_k(x_k)
            epsilon = tf.random.normal(tf.shape(input=std_k))
            z_k = mu_k + self.is_training_ph * epsilon * std_k
            kl = (kl_k if (kl is None) else (kl + kl_k))
            if save_emb:
                z_list.append(z_k)

            # p-network
            z_k = tf.nn.l2_normalize(z_k, axis=1)
            logits_k = tf.matmul(z_k, items, transpose_b=True) / ARG.tau
            probs_k = tf.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))

        logits = tf.math.log(probs)
        logits = tf.nn.log_softmax(logits)
        recon_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(
            input_tensor=-logits * self.input_ph, axis=-1))

        if save_emb:
            return tf.compat.v1.train.Saver(), z_list
        return tf.compat.v1.train.Saver(), logits, recon_loss, kl


def load_data(data_dir):
    pro_dir = os.path.join(data_dir, 'pro_sg')

    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    n_items = len(unique_sid)

    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)

    vad_data_tr, vad_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'validation_tr.csv'),
        os.path.join(pro_dir, 'validation_te.csv'),
        n_items)

    tst_data_tr, tst_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'),
        os.path.join(pro_dir, 'test_te.csv'),
        n_items)
    print(n_items)
    print(train_data.shape[1])
    assert n_items == train_data.shape[1]
    assert n_items == vad_data_tr.shape[1]
    assert n_items == vad_data_te.shape[1]
    assert n_items == tst_data_tr.shape[1]
    assert n_items == tst_data_te.shape[1]

    return (n_items, train_data, vad_data_tr, vad_data_te,
            tst_data_tr, tst_data_te)


def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    #print("rows",rows)
    print("cols",cols.max())
    print("n_users", int(n_users))
    print("n_items",int(n_items))
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float32',
                             shape=(int(n_users), int(n_items)))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']
    
    print("======")
    print( "min uid te" , tp_te['uid'].min())
    print("min uid tr",tp_tr['uid'].min())
    print( "max uid te" , tp_te['uid'].max())
    print( "max uid tr" , tp_tr['uid'].max())

    print("rows_te size" , len(rows_te))
    print("cols_te size" ,len(cols_te))
    print("start_idx", start_idx)
    print("end_idx", end_idx)
    print("n_items",n_items)
    print("size" ,end_idx - start_idx + 1)
    print("******")
    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def load_item_cate(data_dir, num_items):
    assert 'alishop' in data_dir
    data_dir = os.path.join(data_dir, 'pro_sg')

    hash_to_sid = {}
    with open(os.path.join(data_dir, 'unique_sid.txt')) as fin:
        for i, line in enumerate(fin):
            hash_to_sid[int(line)] = i
    assert num_items == len(hash_to_sid)

    hash_to_cid = {}
    with open(os.path.join(data_dir, 'item_cate.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for item, cate in reader:
            item, cate = int(item), int(cate)
            if item not in hash_to_sid:
                continue
            assert item in hash_to_sid
            if cate not in hash_to_cid:
                hash_to_cid[cate] = len(hash_to_cid)
    num_cates = len(hash_to_cid)

    item_cate = np.zeros((num_items, num_cates), dtype=np.bool)
    with open(os.path.join(data_dir, 'item_cate.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for item, cate in reader:
            item, cate = int(item), int(cate)
            if item not in hash_to_sid:
                continue
            item = hash_to_sid[item]
            cate = hash_to_cid[cate]
            item_cate[item, cate] = True
    item_cate = item_cate.astype(np.int64)

    js = np.argsort(item_cate.sum(axis=0))[-7:]
    item_cate = item_cate[:, js]
    assert np.min(np.sum(item_cate, axis=1)) == 1
    assert np.max(np.sum(item_cate, axis=1)) == 1
    return item_cate


def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = x_pred.shape[0]
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0
    return ndcg


def recall_at_k_batch(x_pred, heldout_batch, k=100):
    batch_users = x_pred.shape[0]

    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    recall[np.isnan(recall)] = 0
    return recall


def main_trn(train_data, vad_data_tr, vad_data_te):
    set_rng_seed(ARG.seed)

    n = train_data.shape[0]
    print(n)
    n_items = train_data.shape[1]
    idxlist = list(range(n))

    n_vad = vad_data_tr.shape[0]
    idxlist_vad = list(range(n_vad))

    num_batches = int(np.ceil(float(n) / ARG.batch))
    total_anneal_steps = 5 * num_batches

    tf.compat.v1.reset_default_graph()
    vae = MyVAE(n_items)
    saver, logits_var, train_op_var, merged_var = vae.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_best_var = tf.compat.v1.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.compat.v1.summary.scalar('vad/ndcg', ndcg_var)
    ndcg_best_summary = tf.compat.v1.summary.scalar('vad/ndcg_best', ndcg_best_var)
    merged_valid = tf.compat.v1.summary.merge([ndcg_summary, ndcg_best_summary])

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    summary_writer = tf.compat.v1.summary.FileWriter(LOG_DIR,
                                           graph=tf.compat.v1.get_default_graph())
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    from tqdm import tqdm
    with tf.compat.v1.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        best_ndcg = -np.inf
        update_count = 0.0
        for epoch in range(ARG.epoch):
            print("Epoch ==========> ",epoch)
            np.random.shuffle(idxlist)
            for bnum, st_idx in enumerate(range(0, n, ARG.batch)):
                end_idx = min(st_idx + ARG.batch, n)
                x = train_data[idxlist[st_idx:end_idx]]
                if sparse.isspmatrix(x):
                    x = x.toarray()
                x = x.astype('float32')
                if total_anneal_steps > 0:
                    anneal = min(ARG.beta,
                                 1. * update_count / total_anneal_steps)
                else:
                    anneal = ARG.beta
                feed_dict = {vae.input_ph: x,
                             vae.keep_prob_ph: ARG.keep,
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1}
                sess.run(train_op_var, feed_dict=feed_dict)
                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict=feed_dict)
                    summary_writer.add_summary(
                        summary_train,
                        global_step=epoch * num_batches + bnum)
                update_count += 1

            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, n_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, n_vad)
                x = vad_data_tr[idxlist_vad[st_idx:end_idx]]
                if sparse.isspmatrix(x):
                    x = x.toarray()
                x = x.astype('float32')
                pred_val = sess.run(logits_var, feed_dict={vae.input_ph: x})
                # exclude examples from training and validation (if any)
                pred_val[x.nonzero()] = -np.inf
                ndcg_dist.append(
                    ndcg_binary_at_k_batch(
                        pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg = ndcg_dist.mean()
            print("ndcg===>",ndcg)
            if ndcg > best_ndcg:
                saver.save(sess, '{}/chkpt'.format(LOG_DIR))
                best_ndcg = ndcg
            merged_valid_val = sess.run(
                merged_valid,
                feed_dict={ndcg_var: ndcg, ndcg_best_var: best_ndcg})
            summary_writer.add_summary(merged_valid_val, epoch)

    return best_ndcg


def main_tst(tst_data_tr, tst_data_te, report_r20=False):
    set_rng_seed(ARG.seed)

    n_test = tst_data_tr.shape[0]
    n_items = tst_data_tr.shape[1]
    idxlist_test = list(range(n_test))

    tf.compat.v1.reset_default_graph()
    vae = MyVAE(n_items)
    saver, logits_var, _, _ = vae.build_graph()

    n100_list, r20_list, r50_list = [], [], []
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, '{}/chkpt'.format(LOG_DIR))
        for bnum, st_idx in enumerate(range(0, n_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, n_test)
            x = tst_data_tr[idxlist_test[st_idx:end_idx]]
            if sparse.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: x})
            pred_val[x.nonzero()] = -np.inf
            n100_list.append(ndcg_binary_at_k_batch(
                pred_val, tst_data_te[idxlist_test[st_idx:end_idx]], k=100))
            r20_list.append(recall_at_k_batch(
                pred_val, tst_data_te[idxlist_test[st_idx:end_idx]], k=20))
            r50_list.append(recall_at_k_batch(
                pred_val, tst_data_te[idxlist_test[st_idx:end_idx]], k=50))

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@100=%.5f (%.5f)" % (
        n100_list.mean(), np.std(n100_list) / np.sqrt(len(n100_list))),
          file=sys.stderr)
    print("Test Recall@20=%.5f (%.5f)" % (
        r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))),
          file=sys.stderr)
    print("Test Recall@50=%.5f (%.5f)" % (
        r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))),
          file=sys.stderr)
    if report_r20:
        return r20_list.mean()
    return n100_list.mean()


def main_vis(train_data):
    set_rng_seed(ARG.seed)

    n = train_data.shape[0]
    n_items = train_data.shape[1]
    batch_size_vis = (ARG.batch + ARG.kfac - 1) // ARG.kfac

    tf.compat.v1.reset_default_graph()
    vae = MyVAE(n_items)
    saver, facets_varls, items_var, cores_var = vae.build_graph(save_emb=True)
    with tf.compat.v1.Session() as sess:
        print(LOG_DIR)
        print(sess)
        saver.restore(sess, '{}/chkpt'.format(LOG_DIR))
        items = sess.run(items_var)
        cores = sess.run(cores_var)
        users = []
        for bnum, st_idx in enumerate(range(0, n, batch_size_vis)):
            end_idx = min(st_idx + batch_size_vis, n)
            x = train_data[st_idx:end_idx]
            if sparse.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')
            facets_ls = sess.run(facets_varls, feed_dict={vae.input_ph: x})
            users.append(np.concatenate(facets_ls, axis=1))
        users = np.concatenate(users, axis=0)

    interpretability = 0
    interpretability += visualize_macro(users, items, cores, train_data)
    return interpretability


def np_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm[norm < eps] = eps
    return x / norm


def matchness_of_cores_and_cates(cores, items, cate_items):
    k = cate_items.shape[1]
    cates = np.argmax(cate_items, axis=1)
    cate_centers = []
    for j in range(k):
        cate_centers.append(items[cates == j].sum(axis=0, keepdims=True))
    cate_centers = np.concatenate(cate_centers, axis=0)
    cate_centers = np_normalize(cate_centers)
    core_vs_cate = cores.dot(cate_centers.T)
    print('core_vs_cate =\n', core_vs_cate, file=sys.stderr)
    best_cate_for_core = np.argmax(core_vs_cate, axis=1)
    print('best_cate_for_core = ', best_cate_for_core, file=sys.stderr)
    best_core_for_cate = np.argmax(core_vs_cate, axis=0)
    print('best_core_for_cate = ', best_core_for_cate, file=sys.stderr)
    interpretability = 0
    if len(set(best_core_for_cate)) == k:
        interpretability += 1
    if len(set(best_cate_for_core)) == k:
        interpretability += 1
    if interpretability >= 2:
        inconsistent = False
        for j in range(k):
            if best_core_for_cate[best_cate_for_core[j]] != j:
                inconsistent = True
                break
        if not inconsistent:
            interpretability += 1
    return interpretability, best_cate_for_core


def visualize_macro(users, items, cores, train_data):
    palette = np.asarray(
        [[238, 27., 39., 80],  # _0. Red
         [59., 175, 81., 80],  # _1. Green
         [255, 127, 38., 80],  # _2. Orange
         [255, 129, 190, 80],  # _3. Pink
         [153, 153, 153, 80],  # _4. Gray
         [156, 78., 161, 80],  # _5. Purple
         [35., 126, 181, 80]],  # 6. Blue
        dtype=np.float32) / 255.0

    n, m = users.shape[0], items.shape[0]
    k, d = cores.shape
    users = users.reshape(n, k, d)
    assert items.shape[1] == d
    del n

    users = np_normalize(users)
    items = np_normalize(items)
    cores = np_normalize(cores)

    cate_items = load_item_cate(ARG.data, m)
    interpretable, core2cate = matchness_of_cores_and_cates(
        cores, items, cate_items)
    print('macro interpretable = %d [seed = %d]' % (
        interpretable, ARG.seed),
          file=sys.stderr)
    if interpretable < 3:
        print('Some prototypes do not align well with categories. '
              'Maybe try another random seed? :)', file=sys.stderr)
        return interpretable

    cate_items = cate_items[:, core2cate]  # align categories with prototypes
    gold_items = np.argmax(cate_items, axis=1)
    pred_items = np.argmax(items.dot(cores.T), axis=1)

    ufacs, gold_ufacs = [], []  # user facets
    is_in_ufacs = set()
    for u, i in zip(*train_data.nonzero()):
        c = gold_items[i]
        if (u, c) not in is_in_ufacs:
            is_in_ufacs.add((u, c))
            ufacs.append(users[u, c].reshape(1, d))
            gold_ufacs.append(c)
    del is_in_ufacs
    ufacs = np.concatenate(ufacs, axis=0)
    gold_ufacs = np.asarray(gold_ufacs, dtype=np.int64)
    n = ufacs.shape[0]
    assert ufacs.shape == (n, d)
    assert gold_ufacs.shape == (n,)

    vis_dir = os.path.join(LOG_DIR, 'vis')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)

    def plot(title, xy, color, marksz=2.):
        fig, ax = plt.subplots()
        ax.scatter(x=xy[:, 0], y=xy[:, 1], s=marksz, c=color)
        plt.savefig('%s.png' % title.replace('/', '-'), format='png', dpi=160)

    nodes = np.concatenate((items, ufacs), axis=0)
    assert nodes.shape == (m + n, d)
    gold_nodes = np.concatenate((gold_items, gold_ufacs), axis=0)
    assert gold_nodes.shape == (m + n,)
    pred_nodes = np.argmax(nodes.dot(cores.T), axis=1)

    col_pred = palette[pred_nodes]
    col_gold = palette[gold_nodes]

    tsne2d_sav = os.path.join(LOG_DIR, 'tsne2d-nodes.npy')
    if os.path.isfile(tsne2d_sav):
        x_2d = np.load(tsne2d_sav)
    else:
        x_2d = nodes
        if d > k:
            x_2d = sklearn.decomposition.PCA(
                n_components=k).fit_transform(x_2d)
        print('Running tSNE...', file=sys.stderr)
        x_2d = sklearn.manifold.TSNE(
            n_components=2, perplexity=30).fit_transform(x_2d)
        print('Finished tSNE...', file=sys.stderr)
        np.save(tsne2d_sav, x_2d)
    plot('tsne2d-nodes/pred', x_2d, col_pred)
    plot('tsne2d-nodes/gold', x_2d, col_gold)
    plot('tsne2d-items/pred', x_2d[:m], col_pred[:m])
    plot('tsne2d-items/gold', x_2d[:m], col_gold[:m])
    plot('tsne2d-users/pred', x_2d[m:], col_pred[m:])
    plot('tsne2d-users/gold', x_2d[m:], col_gold[m:])

    return interpretable


def main():
    #tf.config.list_physical_devices('GPU')
    tf.compat.v1.disable_eager_execution()
    (_, train_data, vad_data_tr, vad_data_te,
     tst_data_tr, tst_data_te) = load_data(ARG.data)
    val, tst = 0, 0
    if ARG.mode in ('trn',):
        val = main_trn(train_data, vad_data_tr, vad_data_te)
    if ARG.mode in ('trn', 'tst'):
        tst = main_tst(tst_data_tr, tst_data_te)
        print('(%.5f, %.5f)' % (val, tst))
    if ARG.mode in ('vis',):
        how_interpretable = main_vis(train_data)
        print(how_interpretable)


if __name__ == '__main__':
    main()
