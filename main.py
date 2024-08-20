import warnings
def warn(*arg,**kwargs):pass
warnings.warn= warn

import argparse
import generateFeatures
import predict
# import plot_XAI_figures

import numpy as np


def main(args):
    """
    Main function to process input data, extract features, and predict peptide binding sites.

    Parameters:
    args (Namespace): Command line arguments specifying paths for training and testing datasets, 
                      feature extraction method, and classifier model.
    """
    print('Features extraction begins. Be patient! The machine will take some time.')
    x_train, y_train, train_seq = generateFeatures.process_features(args.train_data_path,args.features.split(','))
    x_test, y_test, test_seq = generateFeatures.process_features(args.test_data_path, args.features.split(','))
        
    print('Preparing prediction file...')
    predict.classifiers(x_train, y_train, x_test, y_test, train_seq, test_seq, args.model, args)
    
    # plot_XAI_figures.classifiers(x_train, y_train, x_test, y_test, train_seq, test_seq, args.model, args)


if __name__ == '__main__':
    """
    Entry point of the script. Parses command line arguments and calls the main function.
    """
    # Create the parser
    p = argparse.ArgumentParser(description='Tool for Peptide Binding Sites Prediction')
    
    # Adding command line arguments for the script
    p.add_argument('-train', '--train_data_path', type=str, help='~/train dataset.csv', default='Datasets/train_dataset.csv')
    p.add_argument('-test', '--test_data_path', type=str, help='~/test dataset.csv', default='Datasets/test_dataset.csv')
    p.add_argument('-f', '--features', type=str, help='List of features separated by commas: MonogramComp, BigramComp, MonogramOccur, BigramOccur', default='MonogramComp')
    p.add_argument('-m', '--model', type=str, help='Classifiers: svm-linear, svm-rbf', default='svm-linear')

    # Parse the arguments
    args = p.parse_args()

    main(args)
