from neo4j import GraphDatabase
import pandas as pd


driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "CAiRE2020neo4j"))

# Delete all nodes and relations in neo4j
def remove_data(tx):
    tx.run("MATCH (a)-[r]-(b) DELETE a,r,b")
    tx.run("MATCH (a) DELETE a")
    
with driver.session() as session:
    session.write_transaction(remove_data)


# Generate CSV file for importing to neo4j
NEO4J_DATA_FOLDER = '/var/lib/neo4j/import' # Check https://neo4j.com/docs/operations-manual/current/configuration/file-locations/
entity_df = pd.read_csv('opendialkg/data/opendialkg_entities.txt', sep='\t', header=None)
entity_df.drop_duplicates(inplace=True)
entity_df.columns = ['entity']
entity_df.to_csv(f'{NEO4J_DATA_FOLDER}/opendialkg_entities.csv',index=False)
print(entity_df)

# Prepare graph triplets
def preprocess_relation(relation):
    relation = relation.replace('.','').replace(',','').replace('/','').replace('(','').replace(')','').replace('\'','')
    relation = relation.replace('  ',' ').replace(' ','_').replace('-','_').replace('~','X_')
    return relation.lower()

triplet_df = pd.read_csv('opendialkg/data/opendialkg_triples.txt', sep='\t',header=None, names=['source','relation','target'])
triplet_df['relation'] = triplet_df['relation'].apply(lambda r: preprocess_relation(r))
triplet_df.drop_duplicates(inplace=True)
triplet_df.to_csv(f'{NEO4J_DATA_FOLDER}/opendialkg_triplet_preprocess.csv', index=False)
print(triplet_df)


# Add entity to neo4j and add unique constraint
def add_entity(tx):
    tx.run("""
        CALL apoc.periodic.iterate('
             load csv with headers from "file:///opendialkg_entities.csv" AS row return row ','
             CREATE (a:Node {value: row.entity})
        ',{batchSize:1000, iterateList:true, parallel:true})
    """)
    
def index_entity(tx):
    tx.run("""
        CREATE CONSTRAINT ON (n:Node) ASSERT n.value IS UNIQUE;
    """)

with driver.session() as session:
    session.write_transaction(add_entity)
    session.write_transaction(index_entity)


# Add relation to neo4j, this step might take some time
def add_relation(tx):
    tx.run("""
        CALL apoc.periodic.iterate('
            load csv with headers from "file:///opendialkg_triplet_preprocess.csv" AS row return row ','
            MATCH (a:Node),(b:Node) 
            WHERE a.value=row.source AND b.value=row.target
            CALL apoc.create.relationship(a, row.relation, {}, b) yield rel
            REMOVE rel.noOp
        ',{batchSize:1000, iterateList:true, parallel:false})""")

with driver.session() as session:
    session.write_transaction(add_relation)



# Check the entity and relation in neo4j
def read_count(tx):
    for record in  tx.run("MATCH (a)-[r]->(b) RETURN COUNT(r)"):
        print(record)
        
with driver.session() as session:
    print(session.read_transaction(read_count))


driver.close()