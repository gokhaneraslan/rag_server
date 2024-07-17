from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from get_embedding_function import MyEmbeddingFunction

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=True)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """

Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):

    embedding_function = MyEmbeddingFunction
    db_chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db_chroma.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=query_text)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    generated_answer = model.generate(input_ids, max_length=54)
    decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
    formated_answer = str(str(decoded_answer).split("<")[1]).split(">")[1]

    print("Answer: ", formated_answer)
    
    return formated_answer

