
from flask import Flask, render_template, send_from_directory, request, jsonify
import requests
from nostril import nonsense
# from interact_adapter import generate, get_perplexity
from utils.translator import translate 
from utils.toxic_detector import get_toxic_score
from models.torchMoji.examples.score_texts_emojis import get_emoji_score
import datetime
import json

app = Flask(__name__)

projectId = 'testaction-d6d53'

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)

@app.route('/models_emo/<path:path>')
def send_models(path):
    return send_from_directory('models_emo', path)
    
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/request', methods=['POST'])
def chat():

    content = request.json
    history = content['history']
    query = content['query']
    task_name = content['task_name']
    lang_id = content['lang_id']
    top_p = content['top_p']
    temp = content['temp']
    rept = content['rept']
    lon = content['lon']
    lat = content['lat']
    
    if(type(history)==str or len(history)==0 or history[0]==""): history = []

    query = translate(query,lang_id,"en",auto=True if task_name=="AUTOMODE" else False)
    emoji_user = get_emoji_score([query]) 

    history.append(query)
    if task_name=="CovidQA": ## COVID QA
        if len(query)<=20 or nonsense(query):
            resp = "Your input doesn't make sense, input something more meaninful"
        else:
            r2 = requests.get('https://covid19api.emos.ai/api/v1/summary/', params={'q': query})
            if(r2!="<Response [200]>"):
                resp = eval(r2.json()[0]['data'])["extractive"]
            else:
                resp = "This service is currenlty not available."
        meta = []
        viz_meta = []
        task_id = 20
    elif task_name=="debunker": ## COVID DEBUNKER
        if len(query)<=20 or nonsense(query):
            resp = "Your input doesn't make sense, input something more meaninful"
        else:
            ppl = get_perplexity(query)
            print(ppl)
            if(float(ppl)> 170.): #### to be tuned
                resp = "This claim is likely to be fake"
            else:
                resp = "This claim is likely to be real"
        meta = []
        viz_meta = []
        task_id = 16
    else:
        resp,meta, task_name, task_id, viz_meta = generate(history=history,task_name=task_name,
                        meta_seed=1,top_p=float(top_p),
                        temperature=float(temp),repetion=float(rept),
                        lon=float(lon), lat=float(lat))

    toxic_ = False
    if task_name == "Neural":
        toxic_resp = get_toxic_score(query,resp)
        if(toxic_resp != ""): 
            resp = toxic_resp
            toxic_ = True

    history.append(resp)
    resp = translate(resp,"en",lang_id, auto=True if task_name=="AUTOMODE" else False)

        
    emoji_resp = get_emoji_score([resp]) 
    jsonFormat = {
        'history': history,
        'response': resp,
        'user_emoji':emoji_user,
        'resp_emoji':emoji_resp,
        'toxic': toxic_,
        'task_name': task_name,
        'task_id': task_id,
        'meta': meta,
        'viz_meta':viz_meta,
        "lon":lon, 
        "lat":lat
    }

    with open("data/conversation_history/"+"{}.json".format(timeStamped("conv")), "w", encoding="utf-8") as f:
        json.dump(jsonFormat,f,indent=4)

    return jsonify(jsonFormat)# last one is to to allow CORS [important]


app.run(host="0.0.0.0", port=8080, ssl_context=('utils/cert.pem', 'utils/key.pem'))


