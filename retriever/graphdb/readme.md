# Installing neo4j
Follow this tutorial to install Neo4J
```
https://jessicawlm.medium.com/installing-neo4j-on-ubuntu-14-04-step-by-step-guide-ed943ec16c56
```
Then install apoc as in: 
```
https://neo4j.com/developer/neo4j-apoc/
```

# Configure Neo4J
For doing a batch loading we need to set a config that allows us to run query longer than 10 sec. 
Then we do the following:
```
sudo vim /etc/neo4j/neo4j.conf
```
search for ```dbms.transaction.timeout=10s``` and modify this number to ```dbms.transaction.timeout=10000000s```. Later after loading the KG, we can put it back to 10s.


# Start Neo4j
Start the Neo4j server. 
```
sudo service neo4j start 
```


# Load the KG into Neo4j
First, use the ```get_data.sh``` to download the data, install neo4j connector for python and give permission to read and write to the neo4j folder. Then we run ```load_db.py```, which will load the KG. 
```
bash get_data.sh
python load_db.py
```

# Query the KG
We have simple script to connect to the graph and run simple queries. 
```
python query.py
```
