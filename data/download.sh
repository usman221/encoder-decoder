#!/usr/bin/env bash

mkdir -p ml-latest-small/pro_sg
wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip

mkdir -p ml-1m/pro_sg
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

mkdir -p ml-20m/pro_sg
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip

