from drqa import retriever
from nltk import tokenize

db = retriever.DocDB()
print("Loading Wikipedia database")
db.__init__('retriever/doc_ret/DrQA/data/wikipedia/docs.db')


def fetch_text(doc_id):
    return db.get_doc_text(doc_id)

print("Loading Ranker index")
ranker = retriever.get_class('tfidf')(tfidf_path='retriever/doc_ret/DrQA/data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')

def get_doc(sentence):
    try:
        doc_names, doc_scores = ranker.closest_docs(sentence, k=1)
        para = fetch_text(doc_names[0]).split("\n")[2]
        para = " ".join(tokenize.sent_tokenize(para)[:1])
        return para, doc_scores
    except:
        return None, [0]

def get_KB_WoW(history):
    try:
        para, doc_scores = get_doc(history[-1])
        print("FIRST DOC")
        print(para)
        print(doc_scores[0])
        if(doc_scores[0]<170):
            para, doc_scores = get_doc(history[-2])
            print("SECOND DOC")

            print(para)
            print(doc_scores[0])
            if(doc_scores[0]<200):
                para, doc_scores = get_doc(history[-3])
                print("THIRD DOC")
                print(para)
                print(doc_scores[0])
                if(doc_scores[0]<170):
                    return [""]
                else:
                    return [para]
            else:
                return [para]
        return [para]
    except:
        return [""]
