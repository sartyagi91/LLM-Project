import os

from easydict import EasyDict as edict


"""
argument parser section to parse arguments to varied functions and classes in the repo
"""


def save_paths():
    
    path_dict={
        'llm_data':'../data/llm_train.csv',
        'model':'ollama/llama3.2'
    }
    
    
    return edict(path_dict)



"""
This section of the code is for passing arguments to model classes,It uses a package called edict which takes in functions returning dictionaries which contain arguments
and then return it as attributes which can be accessed using dot method
"""


# This function returns the embedding model
def model_params():
    
    topicmodel={
        'embeder':'all-MiniLM-L12-v2'
    }

    return edict(topicmodel)

# This function returns model arguments for the classificationmodel.py file for pretraining model
def sent_model():

    modelconfig={
        "model":{'embeder':'all-MiniLM-L6-v2'}
    }

    return edict(modelconfig)

# These are the arguments for the dimensionality reduction model
def dimension_red():
    
    red={
        'umap':{'n_neighbors':15,'n_components':5,
                'metric':'cosine','min_dist':0.0}
    }

    return edict(red)

# These are the arguments for the clustering model

def clustering():

    clus={'hdbscan':{'min_cluster_size':15,
                'metric':'euclidean','cluster_selection_method':'eom',
                'prediction_data':True}}

    clus1= {'hdbscan':{'min_cluster_size':15,
                'metric':'euclidean','cluster_selection_method':'eom',
                'prediction_data':True}}

    return edict(clus), edict(clus1)

# This function returns the arguments for the vectorizer model

def count_vectorizer():
    vec={
        'count':{'stop_words':'english','ngram_range':(1,3),'min_df':10}
    }

    vec1={
        'count':{'stop_words':'english','ngram_range':(1,3)}
    }


    return edict(vec),edict(vec1)



# Function which returns arguments for returning cluster size post topic model training to reduce topics
def finetune_model():

    finetunemodel_p={
        "params":{'cluster_size':30}
    }


    return edict(finetunemodel_p)