from py2neo import Graph, Node, Relationship
import random
import pprint
import json
pp = pprint.PrettyPrinter(indent=2)

# sudo service neo4j start
graph = Graph("http://eez114.ece.ust.hk:7474", auth=("neo4j", "CAiRE2020neo4j"))

def get_global_entity_DIALKG():
    with open('data/dialKG/opendialkg_entities.txt') as f:
        global_entity_list = []
        for x in f:
            global_entity_list.append(x.replace("\n",""))
    return list(set(global_entity_list))
global_ent = get_global_entity_DIALKG()

def _get_KB(entities):
    KB = []
    for ent in entities:
        ent = ent.replace('"',"'")
        results = graph.run(f'MATCH (n1)-[r]->(n2) WHERE n1.value= "{ent}" RETURN n2.value as node,type(r) as rel').data()
        for r in results:
            KB.append(f"{ent}\t{r['rel']}\t{r['node']} ")

        results = graph.run(f'MATCH (n2)-[r]->(n1) WHERE n1.value= "{ent}" RETURN n2.value as node,type(r) as rel').data()
        for r in results:
            KB.append(f"{r['node']}\t{r['rel']}\t{ent}")

    if(len(KB)>5):
        KB = random.sample(KB, 5)
    return KB

def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s.lower() in o.lower() for o in out]):
            out.append(s)
    return out

def _get_ent(utterance):
    temp = []
    for n in global_ent:
        if(n.lower() in utterance.lower()):
            temp.append(n)
    temp = substringSieve(temp)
    temp = list(filter(lambda x: x.isnumeric() or len(x) >= 5, temp))
    temp = list(filter(lambda x: x.lower()!= 'author', temp))
    temp = list(filter(lambda x: x.lower()!= 'no problem', temp))
    temp = list(filter(lambda x: x.lower()!= 'you again', temp))
    temp = list(filter(lambda x: x.lower()!= 'watch', temp))
    temp = list(filter(lambda x: x.lower()!= 'information', temp))
    temp = list(filter(lambda x: x.lower()!= 'haven', temp))
    temp = list(filter(lambda x: x.lower()!= 'review', temp))
    return temp

def get_KB_dialKG(history):
    entities = _get_ent(history[-1])
    if len(entities)==0 and len(history)> 1:
        entities = _get_ent(history[-2])
        if len(entities)==0 and len(history)> 2:
            entities = _get_ent(history[-3])    

    # print(entities)
    KB = _get_KB(entities)
    if(len(KB) == 0):
        return KB, None
    ## VISUALIZATION STUFF
    entities = []
    for k in KB:
        s,r,o = k.split("\t")
        entities.append(s.strip())
        entities.append(o.strip())
    entities = list(set(entities))
    ent_neo = []
    for i_e, e in enumerate(entities):
        ent_neo.append(
            {
                "id": str(i_e),
                "labels": [e],
                "properties": {}
            }
        )

    relation = []
    for ids_x, k in enumerate(KB):
        s,r,o = k.split("\t")
        relation.append({
        "id": str(ids_x),
        "type": r.strip(),
        "startNode": str(entities.index(s.strip())),
        "endNode": str(entities.index(o.strip())),
        "properties": {}
        })

    graph_dic = {
                "results": [{
                    "data": [{
                        "graph": {
                            "nodes": ent_neo,
                            "relationships": relation
                        }
                        }]
                    }],
                "errors": []
                }
    # print(graph_dic)
    graph_dic = json.dumps(graph_dic, indent = 4)
    # pp.pprint(graph_dic)
    # print(graph_dic)
    return KB, graph_dic


# get_KB_dialKG(["Twisted"])