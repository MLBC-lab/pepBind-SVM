import pandas as pd
import itertools
import numpy as np
import ast

def generate_ngram_composition(sequence, n):
    # Generate n-gram features based on recurrence of n successive amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    ngrams = [''.join(ngram) for ngram in itertools.product(amino_acids, repeat=n)]

    ngram_features = [round(sequence.count(ngram) / len(sequence), 3) for ngram in ngrams]

    return ngram_features

def generate_ngram_occurences(sequence, n):
    # Generate n-gram features based on recurrence of n successive amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    ngrams = [''.join(ngram) for ngram in itertools.product(amino_acids, repeat=n)]

    ngram_features = [sequence.count(ngram)  for ngram in ngrams]

    return ngram_features


def generateFeatures(df):
    # Generate monogram features for each protein sequence
    df['MonogramComp'] = df['Sequence'].apply(lambda seq: generate_ngram_composition(seq, 1))

    # Generate bigram features for each protein sequence
    df['BigramComp'] = df['Sequence'].apply(lambda seq: generate_ngram_composition(seq, 2))

    # Generate trigram features for each protein sequence
    # df['TrigramFeatures'] = df['Sequence'].apply(lambda seq: generate_ngram_features(seq, 3))
    df['MonogramOccur'] = df['Sequence'].apply(lambda seq: generate_ngram_occurences(seq, 1))

    # Generate bigram features for each protein sequence
    df['BigramOccur'] = df['Sequence'].apply(lambda seq: generate_ngram_occurences(seq, 2))
    
    return df

def processFeatures(data_path, feature):
  #reading dataset
  data_df = pd.read_csv(data_path)
  data_df = data_df.sample(frac = 1)

  feature_df = generateFeatures(data_df)

  feature_df = data_df[[feature, 'Label']]
  feature_df[feature] = feature_df[feature].apply(np.array)
  feature_df[feature] = feature_df[feature].apply(np.array)

  feature_df = feature_df.reset_index()
  feature_df.drop(columns='index', inplace= True)


  X_train = []
  for i in range(len(feature_df)):
    X_train.append(feature_df[feature].iloc[i])

  y_train = feature_df.reset_index().Label
  y_train = np.array(y_train)

  sequences = data_df.reset_index().Sequence

  return np.array(X_train), y_train, sequences
