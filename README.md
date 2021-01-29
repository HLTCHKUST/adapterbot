## The Adapter-Bot: All-In-One Controllable Conversational Model

<img align="right" src="plot/HKUST.jpg" width="12%">

This is the implementation of the paper:
**The Adapter-Bot: All-In-One Controllable Conversational Model**. [Zhaojiang Lin](https://zlinao.github.io/)*, [Andrea Madotto](https://andreamad8.github.io/)*, Yejin Bang, Pascale Fung  ***AAAI-DEMO*** [[PDF]](https://arxiv.org/pdf/2008.12579.pdf)


## Citation:
If you find this paper and code useful, please cite our paper: 
```
@article{madotto2020adapter,
  title={The Adapter-Bot: All-In-One Controllable Conversational Model},
  author={Madotto, Andrea and Lin, Zhaojiang and Bang, Yejin and Fung, Pascale},
  journal={arXiv preprint arXiv:2008.12579},
  year={2020}
}
```

# Basic Installation
In this repository, we release the trained model, the knowledge retriever, and the interactive script (both via termial and the UI) of the adapter-bot. 

## Download models
To download the pretrained model run the following commands: 
```
## pip install gdown
import gdown
import zipfile
import os


url = 'https://drive.google.com/uc?id=1JQZex-AD-sa5WUT5U4lIn1K2sPW-us8a/'
output = 'models.zip'
gdown.download(url, output, quiet=False)
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall()
os.remove(output)
```

## Download and install knowledge retriever (KG and Wiki)
To download and install the knowledge retrievers you can have to follow the step in the ```retriever``` folder. Specifically, for the knowledge graph follow the read me at:
```
https://github.com/HLTCHKUST/adapterbot/tree/main/retriever/graphdb#installing-neo4j
```
which provides instructions to install neo4j and load opendialoKG. For the wikipedia knowledge, we use [DrQA](https://github.com/facebookresearch/DrQA). Also in this case follow the read me at:
```
https://github.com/HLTCHKUST/adapterbot/tree/main/retriever/doc_ret
```
which provides a simple script for download the wikidump and train the tf-idf retriever.

## Run the interactive script
To interact with the model via command line use the following script:
```
>>> python interact_adapter.py --interact
```

