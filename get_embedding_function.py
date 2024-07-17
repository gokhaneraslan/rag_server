from sentence_transformers import SentenceTransformer


class MyEmbeddingFunction():

  def embed_documents(data):
    model = SentenceTransformer('sentence-transformers/bert-large-nli-stsb-mean-tokens')
    embeddings = model.encode(data)
    return embeddings.tolist()

  def embed_query(query):
    model = SentenceTransformer('sentence-transformers/bert-large-nli-stsb-mean-tokens')
    embeddings = model.encode(query)
    return embeddings.tolist()
