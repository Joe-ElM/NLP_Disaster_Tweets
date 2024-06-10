import string
import re
import spacy
import nltk
import os
import warnings
import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np 
import pandas            as pd 
import missingno         as msno
#import xgboost           as xgb
from spacy.lang.en.examples  import sentences
from nltk.corpus             import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm             import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import make_pipeline
from sklearn.svm             import SVC
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import precision_recall_curve
from sklearn.metrics         import precision_score, recall_score, accuracy_score, f1_score, make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+','', text)

# ---------------------------------------------------------------------

def text_clean(text):
    nlp            = spacy.load('en_core_web_sm')
    StopWords      = stopwords.words('english')
    punctuations   = string.punctuation
    text           = re.sub(r'https?://\S+|www\.\S+','', text)                           # remove URL
    text           = text.replace('&amp;', '')
    doc            = nlp(text)
    lemma_token    = [token.lemma_ for token in doc if token.pos_ != ('PRON' or 'NUM')]  # or'NUM'
    lemma_token_SW = [token.lower() for token in lemma_token if token not in StopWords]
    lemma_token_SW = [token for token in lemma_token_SW if token not in punctuations]
    
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    clean_doc = " ".join(lemma_token_SW)

    # Remove standalone numbers and standalone letters
    clean_doc = re.sub(r'\b\d+\b|\b\w\b', '', clean_doc)

    # Remove characters like 'â', 'ã', 'ª', and punctuation
    final_doc = re.sub(r'[^\w\s]|â|ã|ª', ' ', clean_doc)
    
#     final_doc = re.sub(r'\bvia\b', '', final_doc)   # This line is to be added or deleted to improve score
#     final_doc = re.sub(r'\bnew\b', '', final_doc)   # This line is to be added or deleted to improve scor
    # Tokenize the text using SpaCy
    final_doc = nlp(final_doc)

    # Remove stopwords and non-alphabetic tokens
    final_doc = [token.text for token in final_doc if token.is_alpha and token.text.lower() not in StopWords]
    final_doc = " ".join(final_doc)
    final_doc = re.sub(r'\b\d+\b|\b\w\b', '', final_doc)
    
    return final_doc

# ------------------------------------------------------------------------------------

def keyword_clean(text):
    cleaned_text = re.sub(r'%20', ' ', text)
    return cleaned_text

# -------------------------------------------------------------------------------------


def Confusion_Matrix_Func(y_test, y_pred, model_name):
    """   """
    fig, ax = plt.subplots(figsize=[10, 6])

# -------------------------------------------------------------------------------------------------
    cf_matrix     = confusion_matrix(y_test, y_pred)
    group_names   = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts  = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percent = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percent)]
    labels = np.asarray(labels).reshape(2, 2)

    # ----------------------------------------------------------------------------------------------
    sns.heatmap(cf_matrix,
                annot     = labels,
                ax        = ax,
                annot_kws = {'size': 13},
                cmap      = 'Blues',
                fmt       = ''
                )

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Truth')
    ax.xaxis.set_ticklabels(['Non-Disaster', 'Disaster'])
    ax.yaxis.set_ticklabels(['Non-Disaster', 'Disaster'])
    ax.set_title(f'Confusion Matrix of {model_name}\n')
    
    return cf_matrix, fig