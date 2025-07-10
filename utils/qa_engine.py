from transformers import pipeline

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_with_roberta(question, context_chunks):
    context = " ".join(context_chunks)
    result = qa_model(question=question, context=context)
    return result["answer"]