from py2neo import Graph, Node, Relationship

## pip install py2neo 
# py2neo is a connector to query neo4j

# connect to the KG db
graph = Graph("http://eez114.ece.ust.hk:7474", auth=("neo4j", "CAiRE2020neo4j"))
# graph = Graph("localhost:7687", auth=("neo4j", "CAiRE2020neo4j"))


# run a query
results = graph.run(f'MATCH (n1)-[r]->(n2) WHERE n1.value= "Crazy, Stupid, Love." RETURN n2.value as node,type(r) as rel').data()
print(results)
for r in results:
    print(f"{'Crazy, Stupid, Love.'}\t{r['rel']}\t{r['node']} ")
