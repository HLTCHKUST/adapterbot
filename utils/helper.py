

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size,
            embedding_size=1024,
            load_weight="",
            device='cuda'
    ):
        super(Discriminator, self).__init__()

        self.embed_size = embedding_size
        self.class_size = class_size
        
        self.classifier_head = ClassificationHead(
            class_size=self.class_size,
            embed_size=self.embed_size
        )

        self.classifier_head.load_state_dict(torch.load(load_weight))
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def forward(self, x, label):
        batch_size = x.size(0)
        avg_hidden = x.to(self.device)
        logits = self.classifier_head(avg_hidden)
        label = torch.tensor([label], device='cuda', dtype=torch.long).repeat(batch_size)
        ## LOGGING 
        ce_loss_logging = torch.nn.CrossEntropyLoss(reduction='none')
        loss_logging = ce_loss_logging(logits, label).detach().tolist()
        return loss_logging

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits

def load_classifier(dataset):
    print(f"Loading Classifier {dataset}")
    classifier = None
    class2idx = None
    if dataset == 'sentiments':
        idx2class = ["_", "_", "positive", "negative",
                     "_"]
        models_weight = "data/rankersHead/sentiments/head.pt"
        classifier = Discriminator(
            class_size=len(idx2class),
            embedding_size=1024,
            load_weight=models_weight,
        ).to("cuda")
        classifier.eval()

    elif dataset == 'question':
        idx2class = ["_", "question", "_", "_"]
        models_weight = "data/rankersHead/question/head.pt"
        classifier = Discriminator(
            class_size=len(idx2class),
            embedding_size=1024,
            load_weight=models_weight,
        ).to("cuda")
        classifier.eval()

    elif dataset == "topic":
        idx2class = ["_","sport","business","sci_tech"]
        models_weight = "data/rankersHead/topic/head.pt"
        classifier = Discriminator(
            class_size=len(idx2class),
            embedding_size=1024,
            load_weight=models_weight,
        ).to("cuda")
        classifier.eval()


    elif dataset == "emotion":
        idx2class = ["sadness", "joy", "love", "anger",
                     "fear","surprise"]
        models_weight = "data/rankersHead/emotion/head.pt"
        classifier = Discriminator(
            class_size=len(idx2class),
            embedding_size=1024,
            load_weight=models_weight,
        ).to("cuda")
        classifier.eval()
    class2idx = {c: i for i, c in enumerate(idx2class)}
    return classifier, class2idx


def truncate(f, n):
    if math.isnan(f):
        return f
    return math.floor(f * 10 ** n) / 10 ** n
