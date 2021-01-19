import requests

def f(report):
    toxic = []
    # print(report)
    for k,v in report.items():
        if(v>0.1):
            toxic.append(k)
    # print(toxic)
    return toxic

# docker run -it -p 5000:5000 max-toxic-comment-classifier
def get_toxic_score(query,resp):
    # check for toxicity 
    r2 = requests.post('http://localhost:5000/model/predict', json={'text': [resp]})
    # r2 = requests.post('http://localhost:5000/model/predict', json={'text': [query,resp]})
    # query_toxic = f(eval(r2.text)['results'][0]['predictions'])
    resp_toxic  = f(eval(r2.text)['results'][0]['predictions'])

    str_out = ""
    # if(len(query_toxic)>0):
    #     str_out += f"We detected a possible toxic input, try with some other input please."
    if(len(resp_toxic)>0):
        str_out += f"Our model could have generate a possible offencive response. We are trying our best to detect and correct these cases. Please keep interact with our model."

    return str_out

