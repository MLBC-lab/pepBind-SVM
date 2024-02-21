import pandas as pd
import itertools
import numpy as np
import ast

def generate_ngram_composition(sequence, n):
    """
    Generate n-gram composition for a given sequence.
    
    Args:
        sequence (str): The protein sequence.
        n (int): The length of the n-gram.
        
    Returns:
        ngram_features (list): A list of n-gram composition features.
    """
    # Define the set of amino acids considered in the protein sequences.
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # Generate all possible n-grams from the set of amino acids.
    ngrams = [''.join(ngram) for ngram in itertools.product(amino_acids, repeat=n)]
    
    # Calculate the composition of each n-gram in the sequence.
    ngram_features = [round(sequence.count(ngram) / len(sequence), 3) for ngram in ngrams]

    return ngram_features

def generate_ngram_occurences(sequence, n):
    """
    Generate n-gram occurrences for a given sequence.
    
    Args:
        sequence (str): The protein sequence.
        n (int): The length of the n-gram.
        
    Returns:
        ngram_features (list): A list of n-gram occurrence features.
    """
    # Define the set of amino acids considered in the protein sequences
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # Generate all possible n-grams from the set of amino acids.
    ngrams = [''.join(ngram) for ngram in itertools.product(amino_acids, repeat=n)]

    # Count occurrences of each n-gram in the given sequence.
    ngram_features = [sequence.count(ngram)  for ngram in ngrams]

    return ngram_features


def generateFeatures(df):
    """
    Generate monogram and bigram features for each protein sequence in a DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame containing protein sequences.
        
    Returns:
        df (DataFrame): The DataFrame with added feature columns.
    """
    # Generate monogram composition features
    df['MonogramComp'] = df['Sequence'].apply(lambda seq: generate_ngram_composition(seq, 1))

    # Generate bigram composition features
    df['BigramComp'] = df['Sequence'].apply(lambda seq: generate_ngram_composition(seq, 2))

    # Generate trigram composition features
    # df['TrigramComp'] = df['Sequence'].apply(lambda seq: generate_ngram_composition(seq, 3))

    # Generate monogram occurrence features
    df['MonogramOccur'] = df['Sequence'].apply(lambda seq: generate_ngram_occurences(seq, 1))

    # Generate bigram occurrence features
    df['BigramOccur'] = df['Sequence'].apply(lambda seq: generate_ngram_occurences(seq, 2))
    
    return df

def processFeatures(data_path, feature):
  """
    Process features from a given dataset and a specified feature type.
    
    Args:
        data_path (str): Path to the dataset CSV file.
        feature (str): The type of feature to process (e.g., 'MonogramComp', 'BigramComp').
        
    Returns:
        tuple: A tuple containing the feature array, labels, and sequences.
    """
  # Read dataset and shuffle
  data_df = pd.read_csv(data_path)
  data_df = data_df.sample(frac = 1)

  #generating features
  feature_df = generateFeatures(data_df)

  # Select specified features and labels
  feature_df = data_df[[feature, 'Label']]
  # feature_df[feature] = feature_df[feature].apply(np.array)
  feature_df[feature] = feature_df[feature].apply(np.array)

  feature_df = feature_df.reset_index()
  feature_df.drop(columns='index', inplace= True)

  # Prepare data for training/ testing set
  X_train = []
  for i in range(len(feature_df)):
    X_train.append(feature_df[feature].iloc[i])

  y_train = feature_df.reset_index().Label
  y_train = np.array(y_train)

  sequences = data_df.reset_index().Sequence

  return np.array(X_train), y_train, sequences
