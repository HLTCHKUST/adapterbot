from py2neo import Graph, Node, Relationship

## pip install py2neo 
# py2neo is a connector to query neo4j

# connect to the KG db
graph = Graph("http://eez114.ece.ust.hk:7474", auth=("neo4j", "CAiRE2020neo4j"))
# graph = Graph("localhost:7687", auth=("neo4j", "CAiRE2020neo4j"))


# run a query
results = graph.run("""MATCH (n1)-[r:..*]-(n2) 
                      WHERE n1.value= "Caitlin Moran" 
                      RETURN n1.value as node1, type(r) as rel, n2.value as node2""").data()
for r in results:
    print(f"{r['node1']}\t{r['rel']}\t{r['node2']} ")