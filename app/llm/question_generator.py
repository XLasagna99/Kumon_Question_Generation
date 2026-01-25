def generate_a_question_from_context(context):
    """
    Generate a question based on the provided context.
    
    Inputs:
        context: dict
        {
            "math_problem": dict ({ "operation": str,"a": int,"b": int,"answer": int })
            "context": str # User-input context for generating the question
        }
    
    Outputs:
        question: str # Generated Question Output (first iteration only)

    """
    
    