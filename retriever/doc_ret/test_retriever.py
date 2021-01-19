from drqa import retriever
from nltk import tokenize

db = retriever.DocDB()
print("Loading Wikipedia database")
db.__init__('DrQA/data/wikipedia/docs.db')

print("Loading Ranker index")
ranker = retriever.get_class('tfidf')(tfidf_path='DrQA/data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')

def fetch_text(doc_id):
    return db.get_doc_text(doc_id)



doc_names, doc_scores = ranker.closest_docs("Who is Alan Turing?", k=10)

print(doc_names)
print(doc_scores)

print(fetch_text(doc_names[0]).split("\n"))
