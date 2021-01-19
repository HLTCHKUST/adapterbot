# Document Retriever 
For retrieving documents, we use DrQA. Thus, the first step is to install DrQA as:
```
git clone https://github.com/facebookresearch/DrQA.git
cd DrQA; pip install -r requirements.txt; python setup.py develop
```
The main dependency for DrQA is Stanford CoreNLP. If you do not have it, you can download it here: [https://stanfordnlp.github.io/CoreNLP/index.html#download](https://stanfordnlp.github.io/CoreNLP/index.html#download). And then inside the DrQa folder run: 
```
./install_corenlp.sh
```
Then, you we download the wikipedia article and the Tf-idf matrix direclty from DrQA as:
```
./download.sh
``` 

Finally test your retriever with ```python test_retriever.py```
