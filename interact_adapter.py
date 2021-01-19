import os, random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW,BertTokenizer, BertForSequenceClassification
from caire_adapter import GPT2Adapter
import torch
import argparse
import torch.nn.functional as F
from utils.helper import load_classifier
from utils.dialKG_retriver import get_KB_dialKG
# from utils.WoW_retriver import get_KB_WoW
from utils.WEA_retriver import get_weather
import json
import math
from nltk import tokenize
import urllib.request, json 
import numpy as np

EOS_ID = 50256
TASK_MAP = {"dialGPT":-1,"dialKG":0, "caire":1, "persona":2, "WoW":3, "business":4, 
            "negative":5, "positive":6, "question":7, "sport":8, "sci_tech":9, 
            "SMD":10, "MWoZ_attraction":11, "MWoZ_hotel":12, "MWoZ_restaurant":13, 
            "MWoZ_taxi":14, "MWoZ_train":15, "debunker":16, 
            "anger":17, "fear":18, "joy":19, "sadness":21, "surprised":22, "CovidQA":20}

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
            the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    # assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        logits = logits.squeeze()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        logits = logits.unsqueeze(0)

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def top_k_logits(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def cut_seq_to_eos(sentences,hidden_states):
    hidden_states = torch.cat(hidden_states,1)
    sents = []
    hidden = []
    for id_s, sentence in enumerate(sentences):
        sent=[]
        h_ = []
        for id_t, s in enumerate(sentence):
            if s != EOS_ID:
                sent.append(s)
                h_.append(hidden_states[id_s][id_t])
            else:
                h_ = torch.stack(h_)
                h_ = torch.mean(h_,dim=0)
                hidden.append(h_)    
                break
        sents.append(sent)
    return sents,hidden
    
def sample(model, args, context=None, past=None, device='cuda',
                       sample=True, repetition_penalty=1.2, task_id=-1, classifier=None):
    output = torch.tensor(context, device=device, dtype=torch.long) if context else None
    output_response = output.new_zeros([output.size(0),0])
    stopped = [0 for _ in range(output.size(0))]
    hidden_states = []
    with torch.no_grad():
        for i in range(args.length):

            if past is None and output is not None:
                prev = output[:, -1:]
                _, past = model(output[:, :-1], task_id=task_id) 

        
            logits, past, h = model(prev, past=past, task_id=task_id, get_hidden_states=True)
            hidden_states.append(h)

            logits = logits[:, -1, :] / args.temperature  # + SmallConst
            for i_o, o_ in enumerate(output):
                for token_idx in set(o_.tolist()):
                    if logits[i_o, token_idx] < 0:
                        logits[i_o, token_idx] *= repetition_penalty
                    else:
                        logits[i_o, token_idx] /= repetition_penalty
            # logits = top_k_logits(logits, k=args.top_k)
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)

            if args.no_sample:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            else:
                prev = torch.multinomial(log_probs, num_samples=1)

            output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
            output_response = torch.cat((output_response, prev), dim=1)

            for i_p, p in enumerate(prev.tolist()):
                if(p[0]) == EOS_ID:
                    stopped[i_p] = 1

            if(all(x == 1 for x in stopped)): break
    
    if task_id in [4, 5, 6, 7, 8, 9, 17, 18, 19, 21, 22]:
        output_response, hidden_states = cut_seq_to_eos(output_response,hidden_states)
        # {"sentiments":0,"question":1,"topic":2, "emotion":3}
        mapper = {4:2,5:0,6:0,7:1,8:2,9:2, 17:3, 18:3, 19:3, 21:3, 22:3}
        lable_mapper = {4:2,5:3,6:2,7:1,8:1,9:3, 17:3, 18:4, 19:1, 21:0, 22:5}
        # hidden_states = torch.mean(hidden_states,dim=1)
        hidden_states = torch.stack(hidden_states)
        with torch.no_grad():
            ranks = classifier[mapper[task_id]](hidden_states,lable_mapper[task_id])

        output_response = output_response[argmin(ranks)]
    else:
        output_response = output_response[0]
    return output_response

def automode_taskid(model,token,device):
    task_id = model.classification_HEAD(input_ids = torch.tensor(token, device=device, dtype=torch.long),train=False)[1]
    _, task_id = torch.topk(task_id, k=1, dim=-1)
    task_id = task_id.item()
    return task_id 

def get_rankers():
    classes = {"sentiments":0,"question":1,"topic":2, "emotion":3}
    classifiers = {}
    for discrim, idx in classes.items():
        classifier, _ = load_classifier(discrim)
        classifiers[idx] = classifier.eval()
    return classifiers


def get_perplexity(claim, task_id=16):
    inputs = [EOS_ID] + tokenizer.encode(claim) + [EOS_ID]
    inputs = torch.tensor(inputs, device=args.device, dtype=torch.long)
    lm_logits, *_ = model(input_ids=inputs, task_id=task_id)
    lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
    lm_labels_flat_shifted = inputs[..., 1:].contiguous().view(-1)
    l = loss(lm_logits_flat_shifted, lm_labels_flat_shifted)
    return math.exp(l.item())

parser = argparse.ArgumentParser()
parser.add_argument('--repetition_penalty', type=float, default=1.3)
parser.add_argument("--model_checkpoint", type=str, default="models/adapterbot", help="Path, url or short name of the model")
parser.add_argument("--model_checkpoint_task_id", type=str, default="models/taskselector", help="Path, url or short name of the model")
parser.add_argument('--length', type=int, default=50, help='max generation length')
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument("--top_p", type=float, default=0.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--max_history', type=int, default=2)
parser.add_argument("--task", type=str, default="all", help="choose one from [dialKG, caire, persona, WoW, business, negative, positive, question, sport, sci_tech, SMD, MWoZ_attraction, MWoZ_hotel, MWoZ_restaurant, MWoZ_taxi, MWoZ_train]")
parser.add_argument('--task_id', type=int, default=-1)
parser.add_argument('--temperature', type=float, default=1.0, help='')
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
parser.add_argument('--no_sample', action='store_true')
parser.add_argument("--eval", action='store_true')
parser.add_argument("--interact", action='store_true')
parser.add_argument("--use_history_task_selector", action='store_true')
args = parser.parse_args()



torch.manual_seed(777)
torch.random.manual_seed(777)
torch.cuda.manual_seed(777)
random.seed(777)

classifiers = get_rankers()
loss = torch.nn.CrossEntropyLoss()
tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
model = GPT2Adapter.from_pretrained(args.model_checkpoint)
model.to(args.device)

tokenizer_task_id = BertTokenizer.from_pretrained(args.model_checkpoint_task_id)
model_task_id = BertForSequenceClassification.from_pretrained(args.model_checkpoint_task_id)
model_task_id.to(args.device)


# get caire persona
caire_persona = [
    "my name is caire",
    "i am an empathetic chatbot",
    "i want to help human to make a better world",
    "i am an good friend of human"
]
caire_persona = sum([tokenizer.encode(m) for m in caire_persona],[])

# get random persona
with open("data/persona/test.json", encoding="utf-8") as f:
    personachat = json.load(f)
random_personas = []
for i, dial in enumerate(personachat):
    random_personas.append(sum([tokenizer.encode(m) for m in dial['meta']],[]))
    if(i==50): break




INVERT_TASK_MAP = {int(c): i for i, c in TASK_MAP.items()}
# TASK_MAP[-1] = "neutral"
def generate(history,task_name=-1,meta_seed=1, top_p=0.9, temperature=1.0,repetion=1.2,lon=0,lat=0):
    #show meta in the user interface
    if task_name=="AUTOMODE":
        ## BERT
        if(args.use_history_task_selector):
            dial_history = "[SEP]".join(history[-(2*args.max_history+1):])
        else:
            dial_history = history[-1]

        dial_history = [tokenizer_task_id.encode(dial_history,add_special_tokens=True)]
        input_ids_tok = torch.tensor(dial_history, device=args.device, dtype=torch.long)
        with torch.no_grad():
            outputs = model_task_id(input_ids=input_ids_tok)[0]
        task_id = torch.topk(outputs, k=1, dim=-1)[1].squeeze().item()
        mapper_task = {0:0, 1:1, 2:2, 3:3, 4:random.choice([-1,4,5,6,7,8,9,17,18,19,21,22]),
                        5:10,6:10,7:10,8:11,9:12,10:13,11:14,12:15}
        task_id = mapper_task[task_id]
        print("SELECTED TASK ID",INVERT_TASK_MAP[task_id])
        task_name = INVERT_TASK_MAP[task_id]

    history = history[-(2*args.max_history+1):]
    context_tokens = sum([tokenizer.encode(h) + [EOS_ID] for h in history],[])
    meta = []
    viz_meta = {"graph":None,"Wiki":None,"Wea":None}
    if(task_name in ['Movie',"Book","Music","Sport"] or task_name == "dialKG"):
        KB, graph_dic = get_KB_dialKG(history)
        viz_meta["graph"] = graph_dic
        meta = sum([tokenizer.encode(m) for m in KB],[])
        task_name = "dialKG"
    if task_name == "caire":
        meta = caire_persona
    if task_name == "persona":
        meta = random_personas[meta_seed]
    if(task_name == "WoW"):
        KB = get_KB_WoW(history)
        meta = sum([tokenizer.encode(m) for m in KB],[])
        viz_meta["Wiki"] = KB

    if(task_name in ['schedule',"navigate","weather","SMD"]):
        # if(task_name == "weather"):
        meta_w, dict_vix, err_MSG = get_weather(history[-1],lon,lat)
        viz_meta["Wea"] = dict_vix
        meta = sum([tokenizer.encode(m) for m in meta_w],[]) 
        task_name = "SMD"
        
    if(task_name in ["anger","fear","joy","sadness",
                   "surprised","positive","negative",
                   "question","business","sport","sci_tech"]):
        args.num_samples = 10
        args.top_k = 10
        args.top_p = 0.0
        args.temperature = temperature
        args.repetition_penalty = 1.3
    elif(task_name in ["MWoZ_attraction","MWoZ_taxi","MWoZ_hotel","MWoZ_restaurant","MWoZ_train","SMD"]):  
        args.num_samples = 1
        args.top_k = 1
        args.top_p = 0.0
        args.temperature = 1.0
        args.repetition_penalty = 1.0
    elif task_name == "dialKG":
        args.num_samples = 1
        args.top_k = 1
        args.top_p = 0.0
        args.temperature = temperature
        args.repetition_penalty = repetion  
    else:
        args.num_samples = 1
        args.top_k = 0
        args.top_p = top_p
        args.temperature = temperature
        args.repetition_penalty = repetion

    if len(meta)>0:
        context_tokens = meta + [EOS_ID] + context_tokens

    context_tokens = [context_tokens for _ in range(args.num_samples)]
    task_id = TASK_MAP[task_name]
    original_sentence = sample(model=model,args=args, context=context_tokens, device=args.device,
                               repetition_penalty=args.repetition_penalty, task_id=task_id,
                               classifier=classifiers)
    text = tokenizer.decode(original_sentence, skip_special_tokens=True)
    
    if task_name in ["anger","fear","joy","sadness",
                   "surprised","positive","negative",
                   "question","business","sport","sci_tech"]:
        text = " ".join(tokenize.sent_tokenize(text)[:2])

    return text, tokenizer.decode(meta, skip_special_tokens=True), task_name, task_id, viz_meta

def evaluate(task):
    save_result = os.path.join("data", "results", task)
    test_path = os.path.join("data", task, "test.json")

    with open(test_path, encoding="utf-8") as f:
        data = json.load(f)
    if task in ["caire", "persona", "WoW"]:
        results = []
        for i, dial in enumerate(data):
            result = {"meta":[],"dialogue":[]}
            result["meta"] = dial["meta"]
            history = []
            meta = sum([tokenizer.encode(m) for m in dial['meta']],[])
            for turn in dial['dialogue']:
                history.append( tokenizer.encode(turn[0]) )
                context_tokens = sum([tokenizer.encode(h) + [EOS_ID] for h in history[-(2*args.max_history+1):]],[])
                if len(meta)>0:
                    context_tokens = meta + [EOS_ID] + context_tokens
                context_tokens = [context_tokens for _ in range(args.num_samples)]
                original_sentence = sample(model=model,args=args, context=context_tokens, device=args.device,
                                        repetition_penalty=args.repetition_penalty, task_id=TASK_MAP[task],
                                        classifier=classifiers)
                text = tokenizer.decode(original_sentence, skip_special_tokens=True)
                result["dialogue"].append([text])
                history.append(tokenizer.encode(turn[1]))
            results.append(result)
    
    elif task in ["SMD"]:
        results = {}
        for i, dial in enumerate(data):
            results[str(i)] = []
            history = []
            meta = sum([tokenizer.encode(m) for m in dial['meta']],[])
            for turn in dial['dialogue']:
                history.append( tokenizer.encode(turn[0]) )
                context_tokens = sum([tokenizer.encode(h) + [EOS_ID] for h in history[-(2*args.max_history+1):]],[])

                if len(meta)>0:
                    context_tokens = meta + [EOS_ID] + context_tokens
                context_tokens = [context_tokens for _ in range(args.num_samples)]
                original_sentence = sample(model=model,args=args, context=context_tokens, device=args.device,
                                        repetition_penalty=args.repetition_penalty, task_id=TASK_MAP[task],
                                        classifier=classifiers)
                text = tokenizer.decode(original_sentence, skip_special_tokens=True)
                results[str(i)].append({"spk":"SYS", "text":text})
                history.append(tokenizer.encode(turn[1]))
    elif task in ["dialKG"]:
        results = {}
        for dial in data:
            if str(dial["id"]) not in results:
                results[str(dial["id"])] = []

            meta = sum([tokenizer.encode(m) for m in dial['meta']],[])
            for turn in dial['dialogue']:
                history = [tokenizer.encode(turn[0])]
                context_tokens = sum([tokenizer.encode(h) + [EOS_ID] for h in history[-(2*args.max_history+1):]],[])
                if len(meta)>0:
                    context_tokens = meta + [EOS_ID] + context_tokens
                context_tokens = [context_tokens for _ in range(args.num_samples)]
                original_sentence = sample(model=model,args=args, context=context_tokens, device=args.device,
                                        repetition_penalty=args.repetition_penalty, task_id=TASK_MAP[task],
                                        classifier=classifiers)
                text = tokenizer.decode(original_sentence, skip_special_tokens=True)
                results[str(dial["id"])].append({"spk":"SYS", "text":text})
    elif "MWoZ" in task:
        results = {}
        for i, dial in enumerate(data):
            results[dial["id"]] = []
            history = []
            meta = sum([tokenizer.encode(m) for m in dial['meta']],[])
            for turn in dial['dialogue']:
                history.append( tokenizer.encode(turn[0]) )
                context_tokens = sum([tokenizer.encode(h) + [EOS_ID] for h in history[-(2*args.max_history+1):]],[])

                if len(meta)>0:
                    context_tokens = meta + [EOS_ID] + context_tokens
                context_tokens = [context_tokens for _ in range(args.num_samples)]
                original_sentence = sample(model=model,args=args, context=context_tokens, device=args.device,
                                        repetition_penalty=args.repetition_penalty, task_id=TASK_MAP[task],
                                        classifier=classifiers)
                text = tokenizer.decode(original_sentence, skip_special_tokens=True)
                if "=" in turn[1]:
                    results[dial["id"]].append({"spk":"SYS-API", "text":text})
                else:
                    results[dial["id"]].append({"spk":"SYS", "text":text})
                history.append(tokenizer.encode(turn[1]))
    if not os.path.exists(save_result):
        os.makedirs(save_result)
    with open(os.path.join(save_result, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results,f,indent=4)


if args.eval:
    tasks = ["caire", "persona", "dialKG", "MWoZ_attraction", "MWoZ_hotel", "MWoZ_restaurant", "MWoZ_taxi", "MWoZ_train", "SMD", "WoW"]
    tasks = ["dialKG"]
    for task in tasks:
        evaluate(task)
if args.interact:  
    tasks_options = {"AUTOMODE":"AUTOMODE","a": "dialGPT","b": "dialKG","c": "caire","d": "persona","e": "WoW","f": "business","g": "negative","h": "positive","i": "question","l": "sport","m": "sci_tech","n": "SMD","o": "anger","p": "fear","q": "joy","r": "sadness","s": "surprised"}
    print("AdapterBot provides AUTO and MANUAL mode.")
    print("Please, select the mode you prefer by typing AUTO or MANUAL:")
    setting = input("MODE >>>")
    while setting not in ["AUTO","MANUAL"]:
        setting = input("MODE >>>")
    if setting == "AUTO": print("The model select automatically which dialogue skill to use.")
    elif setting == "MANUAL": print("At each turn you have to select the dialogue skill manually.")
    print()
    print()
    print()
    history = []
    while True:
        raw_text = input("USR >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("USR >>>")
        history.append(raw_text)
        if(setting =="MANUAL"):
            print("Select your task id between: ")
            print("(a) dialGPT (b) dialKG (c) caire (d) persona (e) WoW (f) business (g) negative (h) positive (i) question (l) sport (m) sci_tech (n) SMD (o) anger (p) fear (q) joy (r) sadness (s) surprised")
            task_id = input("TASK_ID >>> ")
            while task_id not in tasks_options.keys():
                task_id = input("TASK_ID >>> ")
        else: task_id = "AUTOMODE"
        text, _, _, _, _  = generate(history, tasks_options[task_id])

        print(f"SYS >>> {text}")
        history.append(text) 
