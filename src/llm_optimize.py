import random
from tqdm import tqdm
import pandas as pd

from metrics import *
from signature import *
            
def sample_data(path:str,num_samps:int):
    random.seed(42)
    sample_idx=[]
    df=pd.read_csv(path)
    labels=dict(df.groupby(by='perspective').apply(len).astype(int))

    for l in labels:
        idx_list=df[df['perspective']==l].index.to_list()
        
        sample_idx.extend(random.sample(idx_list,num_samps))

        
    return df.loc[sample_idx].reset_index(drop=True)

def inference(trainset,n: int):
    scores = []
    model=Model()
    metric_summary=SemanticF1_Summary(decompositional=False)

    for x in tqdm(trainset[:n]):
        try:
            pred = model(**x.inputs())
            scores.append(metric_summary(x,pred))
        except:
            scores.append('None')
            
    return scores

class Model(dspy.Module):
    def __init__(self):
        super().__init__()
        self.llm_summary=dspy.ChainOfThought(Summary)
        
    def forward(self,question,answers,perspective,
                perspective_definition,tone_attribute):
        summary=self.llm_summary(question=question,
                                answers=answers,
                                perspective=perspective,
                                perspective_definition=perspective_definition,
                                tone_attribute=tone_attribute,
                                )
        
        return summary
    
    
if __name__=="__main__":
    llm = dspy.LM(model='ollama/llama3.2',max_tokens=10000)
    dspy.configure(lm=llm)