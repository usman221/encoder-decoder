#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

# Script for converting the netflix dataset into the format accepted by prep.py.

DATA_DIR = './netflix/'

start_total = datetime.now()
savfile = os.path.join(DATA_DIR, 'ratings.csv')
if not os.path.isfile(savfile):
    data = open(savfile, 'w')
    data.write('userId,movieId,rating,timestamp\n')
    files_to_be_read = ['combined_data_1.txt', 'combined_data_2.txt',
                        'combined_data_3.txt', 'combined_data_4.txt']
    for file in files_to_be_read:
        start_individual = datetime.now()
        with open(os.path.join(DATA_DIR, file)) as opened_file:
            for line in opened_file:
                line = line.strip()
                if line.endswith(':'):
                    m_id = line.replace(':', '')
                else:
                    row = [x for x in line.split(',')]
                    row.insert(1, m_id)
                    data.write(','.join(row))
                    data.write('\n')
            end_individual = datetime.now()
            print('done with {} file in {}'.format(file, (
                    end_individual - start_individual)))
    data.close()
end_total = datetime.now()
print("total Time = {}".format(end_total - start_total))
