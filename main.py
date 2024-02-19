import warnings
def warn(*arg,**kwargs):pass
warnings.warn= warn

import argparse
import generateFeatures
import predict

import numpy as np


def main(args):

    print('Features extraction begins. Be patient! The machine will take some time.')
    x_train, y_train, train_seq = generateFeatures.processFeatures(args.train_data_path,args.feature)
    x_test, y_test, test_seq = generateFeatures.processFeatures(args.test_data_path, args.feature)
    
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    print('Preparing prediction file...')
    predict.classifiers(x_train, y_train, x_test, y_test, train_seq, test_seq, args.model, args)
    
    

if __name__ == '__main__':
    ######################
    # Adding Arguments
    #####################

    p = argparse.ArgumentParser(description='Tool for Peptide Binding Sites Prediction')
    
    p.add_argument('-train', '--train_data_path', type=str, help='~/train dataset.csv', default='Datasets/train_data.csv')
    p.add_argument('-test', '--test_data_path', type=str, help='~/test dataset.csv', default='Datasets/test_data_200_200.csv')
    p.add_argument('-fe', '--feature', type=str, help='Feature: MonogramComp/BigramComp/MonogramOccur/BigramOccur', default='MonogramOccur')
    p.add_argument('-m', '--model', type=str, help='Classifiers: svm-linear, svm-rbf', default='svm-linear')
    args = p.parse_args()

    main(args)