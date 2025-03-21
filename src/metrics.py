import dspy


from rouge_score import rouge_scorer

def evaluation(ground_truth:str,pred:str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth,pred)

    return scores


class DecompositionalSemanticRecallPrecision(dspy.Signature):
    """
    Compare a system's response to the ground truth to compute recall and precision of key ideas.
    You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision.
    """

    question: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    ground_truth_key_ideas: str = dspy.OutputField(desc="enumeration of key ideas in the ground truth")
    system_response_key_ideas: str = dspy.OutputField(desc="enumeration of key ideas in the system response")
    discussion: str = dspy.OutputField(desc="discussion of the overlap between ground truth and system response")
    recall: float = dspy.OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = dspy.OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")

class SemanticRecallPrecision(dspy.Signature):
    """
    Compare a system's response to the ground truth to compute its recall and precision.
    If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.
    """

    question: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    
    recall: float = dspy.OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = dspy.OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")

def f1_score(precision, recall):
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1_Summary(dspy.Module):
    def __init__(self, threshold=0.65, decompositional=False):
        self.threshold = threshold

        if decompositional:
            self.module = dspy.ChainOfThought(DecompositionalSemanticRecallPrecision)
        else:
            self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def forward(self, example, pred, trace=None):
        
        try:
            scores = self.module(question=example.question, ground_truth=example.summaries, system_response=pred.gen_summary)
            score = f1_score(scores.precision, scores.recall)

        except:
            score=0
                
        return score if trace is None else score >= self.threshold
