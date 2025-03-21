import dspy



class Summary(dspy.Signature):
    """
    Summarize the following content according to the perspective and the asked question. The abstractive summary created should detect the spans in the different answers describing the particular perspective. Utilize the definition of the perspective and the tone information for accurate summary generation.
    """
    question: str = dspy.InputField(desc="The question")
    answers: str = dspy.InputField(desc="The answers to the question asked which needs to be summarized")
    
    perspective: str = dspy.InputField(desc="The Perspective for which the summary needs to attuned to")
    
    perspective_definition: str = dspy.InputField(desc="defines the semantic of the particular perspective that helps the model understand the specific medical context")
    
    tone_attribute: str = dspy.InputField(desc="The tone reflects the stylistic approach the summary should take.")
    
    gen_summary: str= dspy.OutputField(desc="The generated summary")