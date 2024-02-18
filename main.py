import warnings
def warn(*arg,**kwargs):pass
warnings.warn= warn

import argparse
import generateFeatures


import numpy as np


def main(args):

    print('Features extraction begins. Be patient! The machine will take some time.')
    x_train, y_train = generateFeatures.processFeatures(args.train_data_path,args.feature)
    x_test, y_test = generateFeatures.processFeatures(args.test_data_path, args.feature)
    
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print('Preparing prediction file...')
    

if __name__ == '__main__':
    ######################
    # Adding Arguments
    #####################

    p = argparse.ArgumentParser(description='Feature Geneation Tool from Peptide Sequences')
    
    p.add_argument('-train', '--train_data_path', type=str, help='~/FASTA.txt', default='Datasets/train_data.csv')
    p.add_argument('-test', '--test_data_path', type=str, help='~/FASTA.txt', default='Datasets/test_data_200_200.csv')
    p.add_argument('-fe', '--feature', type=str, help='~/FASTA.txt', default='MonogramOccur')

    args = p.parse_args()

    main(args)