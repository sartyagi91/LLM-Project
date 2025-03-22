import pandas as pd
from collections import defaultdict
import argparse

from bertopic import BERTopic
from sklearn.feature_extraction.text import  CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN


from dataloader import *
from config import *
from args import * 


def model(train_path:str,
        save_model:bool,
        ckpt_path:str):
    
    l_sugg =  ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding","Prescriptive"]
    l_exp = ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful", "Emotional"]
    l_info =  ["Clinical", "Scientific","Informative", "Educational","Factual", "Informing","Academic","Analytical"]
    l_cause =  ["Diagnostic", "Explanatory", "Causal","Due to", "Resulting from", "Attributable to" ]
    l_qs =  ["Inquiry", "Rhetorical", "Exploratory Questioning", "Clarifying Inquiry", "Problem-Solving Deliberation"]
    
    seed_topic_list=[l_sugg,l_exp,l_info,l_cause,l_qs]
    
    modelparams,dimred,(cluster,cluster2),(vectorizer,vectorizer2),finetuneModel=model_params(),dimension_red(),clustering(),count_vectorizer(),finetune_model()


    topic_df=topic_dataset(train_path)

    model=SentenceTransformer(modelparams.embeder)

    umap_model=UMAP(n_neighbors=dimred.umap.n_neighbors,
                    n_components=dimred.umap.n_components,metric=dimred.umap.metric,
                    min_dist=dimred.umap.min_dist)

    hdbscan_model=HDBSCAN(min_cluster_size=cluster2.hdbscan.min_cluster_size,
                        metric=cluster2.hdbscan.metric,cluster_selection_method=cluster2.hdbscan.cluster_selection_method,
                        prediction_data=cluster2.hdbscan.prediction_data)

    vectorizer_model=CountVectorizer(stop_words=vectorizer2.count.stop_words,
                                    ngram_range=vectorizer2.count.ngram_range)



    ctfidf_model = ClassTfidfTransformer()


    # Adding Maximum Marginal Relevance score to add keywords from documents which are diverse
    # and remove repeated keywords

    rep_model=MaximalMarginalRelevance(diversity=0.9)


    topic_model = BERTopic(
        calculate_probabilities=True,
        seed_topic_list=seed_topic_list,
        embedding_model=model,          
        umap_model=umap_model,                    
        hdbscan_model=hdbscan_model,
        ctfidf_model=ctfidf_model,              
        vectorizer_model=vectorizer_model,
        representation_model=rep_model
        )
    
    topics,probs=topic_model.fit_transform(documents=topic_df['answers'])
    
    if save_model:
        topic_model.save(ckpt_path,serialization="safetensors", save_ctfidf=True)
        
        
    else:
        BERTopic.load(ckpt_path)


if __name__=="__main__":

    args=topic_model_args()
    model(args.data_dir,args.save_model,args.checkpoint_dir)