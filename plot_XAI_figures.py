import numpy as np
import pandas as pd
import shap
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, \
        confusion_matrix, auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef

def get_features_names():
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    # Generate all possible n-grams from the set of amino acids.
    ngrams_occur = [''.join(ngram) + '_occur' for ngram in itertools.product(amino_acids, repeat=1)]
    ngrams_comp = [''.join(ngram) + '_Comp' for ngram in itertools.product(amino_acids, repeat=1)]
    ngrams_occur_2 = [''.join(ngram) + '_occur' for ngram in itertools.product(amino_acids, repeat=2)]
    ngrams_comp_2 = [''.join(ngram) + '_Comp' for ngram in itertools.product(amino_acids, repeat=2)]
    # return ngrams_occur + ngrams_comp + ngrams_occur_2 + ngrams_comp_2
    return ngrams_occur

def classifiers(x_train, y_train, x_test,  y_test, train_seq, test_seq, model, args):
    """
    Train a classifier based on the specified model and evaluate its performance on a test set.
    
    Args:
        x_train (numpy array): Training features.
        y_train (numpy array): Training labels.
        x_test (numpy array): Testing features.
        y_test (numpy array): Testing labels.
        train_seq (Series): Training sequences (not used in training, included for potential future use).
        test_seq (Series): Testing sequences for output file.
        model (str): Model type ('svm-rbf' or 'svm-linear').
        args (Namespace): Additional arguments, not used in this function but included for future extensions.
    
    Outputs:
        Writes the model predictions and true values to 'output.csv'.
        Prints the evaluation report including various performance metrics.
    """
    # Initialize lists to store performance metrics
    accuracy, F1_score, AUC, MCC, recall = 0, 0, 0, 0, 0
    CM = np.zeros((2, 2), dtype=int)

    # Model selection based on the specified model type
    if model == 'svm-rbf':
        model1 = SVC(probability=True)
    elif model == 'svm-linear':
        model1 = SVC(kernel= 'linear', probability=True)

    save_figures = True
    feature_names = get_features_names()
    x_train_df = pd.DataFrame(x_train, columns=feature_names)
    x_test_df = pd.DataFrame(x_test, columns=feature_names)

    # Training and inference
    model1.fit(x_train_df, y_train)
    y_pred = model1.predict(x_test_df)
    y_proba = model1.predict_proba(x_test_df)[:, 1]

    # Calculating performance metrics
    CM += confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred) * 100
    F1_score = f1_score(y_test, y_pred)
    MCC = matthews_corrcoef(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    FPR, TPR, _ = roc_curve(y_test, y_proba)
    AUC = auc(FPR, TPR)

    # Writing prediction to a csv file
    true_val = ["Positive" if label == 1 else "Negative" for label in y_test]
    pred_val = ["Positive" if pred == 1 else "Negative" for pred in y_pred]

    # df = pd.DataFrame({'Sequences': test_seq, 'True Value': true_val, 'Prediction': pred_val})
    # df.to_csv('output.csv',index=False)

    # Print performance metrics
    print('{} Evaluation Report {}'.format(' -'*25,' -'*25))
    TN, FP, FN, TP = CM.ravel()
    print('True Negative: {}, False Positive: {}, False Negative: {}, True Positive: {}'.format(TN, FP,FN,TP))
    print('Accuracy: {0:.2f}%'.format(accuracy))
    print('Sensitivity: {0:.2f}%'.format(((TP) / (TP + FN)) * 100.0))
    print('Specificity: {0:.2f}%'.format(((TN) / (TN + FP)) * 100.0))
    print('F1_Score: {0:.2f}'.format(F1_score))
    print('MCC: {0:.2f}'.format(MCC))
    print('AUC: {0:.2f}'.format(AUC))


    # SHAP explanation
    explainer = shap.Explainer(model1, x_train)
    shap_values = explainer(x_test_df)
  
    # Waterfall plot
    for i in [0,1,2,3,4,5,50,100,150,200,250,300]:
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[i], max_display=20, show=False)
        if save_figures:
            filename = 'XAI Figures_monoOccur/waterfall_plot_'+str(i)+'.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_test, plot_type="violin", feature_names=feature_names, show=False)
    if save_figures:
        plt.savefig('XAI Figures_monoOccur/beeswarm_plot.png', dpi=300, bbox_inches='tight')
    
    #Global feature importance plot
    # Extract SHAP values from Explanation object
    shap_values_array = shap_values.values

    # Convert to numpy array if needed
    if not isinstance(shap_values_array, np.ndarray):
        shap_values_array = np.array(shap_values_array)

    # Calculate global feature importance
    global_importance = np.abs(shap_values_array).mean(axis=0)

    # Sort features by importance and get the indices of the top N features
    top_features_indices = np.argsort(global_importance)[::-1][:20]  # Select top 10 features

    # Get the importance of the top N features and the importance of the rest
    top_features_importance = global_importance[top_features_indices]
    other_features_importance = global_importance.sum() - top_features_importance.sum()

    # Names of top features
    top_feature_names = [feature_names[i] for i in top_features_indices]

    # Plot global feature importance
    plt.figure(figsize=(10, 6))
    # Get a colormap
    cmap = plt.get_cmap('viridis')
    # Plot top N features
    # plt.barh(range(len(top_feature_names)), top_features_importance, color='blue', tick_label=top_feature_names)
    # Plot top N features with different colors
    colors = [cmap(i / len(top_feature_names)) for i in range(len(top_feature_names))]
    plt.barh(range(len(top_feature_names)), top_features_importance, color=colors, tick_label=top_feature_names)

    # Plot the rest of the features as "Other"
    # plt.barh(len(top_feature_names), other_features_importance, color='gray', label='Other')

    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.title('Global Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to display features from top to bottom
    # plt.legend()
    plt.tight_layout()


    # Save the figure
    if save_figures:
        plt.savefig('XAI Figures_monoOccur/global_feature_importance_2.png', dpi=300, bbox_inches='tight')
    for i in [0,1,2,3,4,5,50,100,150,200,250,300]:
        print(i, y_test[i], y_proba[i])