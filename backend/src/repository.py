import time
import gridfs
from pymongo import MongoClient
from rdflib import Graph
import config
import services

## TODO __init__ method

## MongoDB database conn
client = MongoClient(config.MONGO_URI)
db = client[config.MONDO_DB]
DISEASES_COLLECTION = db['diseases']
DATA_MODEL_COLLECTION = db['data_model']
PHENOTYPES_COLLECTION = db['phenotypes']
ANATOMICAL_COLLECTION = db['anatomical']
RO_COLLECTION = db['relationships']

# FileGrid conn
fs = gridfs.GridFS(db)

HPO_KG = Graph()
def get_hpo_kg():
    file = fs.find_one({"filename": "ro.ttl"})
    if file is None:
        raise Exception(f"File ro.ttl not found in MongoDB GridFS.")
    
    ttl_data = file.read().decode('utf-8')

    graph = Graph()
    graph.parse(data=ttl_data, format='xml')
    return graph

# TODO mongoDB adapter
# TODO DynamoDB adapter

# TODO move to repository, define interface
# diseases collection desde MongoDB
def get_diseases():
    diseases = DISEASES_COLLECTION.find({})
    return list(diseases)

def get_diseases_ids():
    """
    Fetch all disease IDs from the DISEASES collection.
    
    Returns:
    list: A list of all disease IDs.
    """
    return DISEASES_COLLECTION.distinct("id")

def get_diseases_by_ids(filtered_disease_ids):
    """
    Fetch all disease from the DISEASES collection for the given list of ids
    
    Returns:
    list: A list of all disease IDs.
    """
    return list(DISEASES_COLLECTION.find({"id": {"$in": list(filtered_disease_ids)}}, {'_id': 0}))


# diseases collection desde MongoDB
def get_data_model():
    data_model = DATA_MODEL_COLLECTION.find_one({}, {'_id': 0})
    return data_model

# get disease by id
def get_disease_by_id(full_id):
    diseases = DISEASES_COLLECTION.find_one({"id": full_id}, {'_id': 0})
    return diseases

# get phenotype by id
def get_phenotype_by_id(full_id):
    phenotype = PHENOTYPES_COLLECTION.find_one({"id": full_id}, {'_id': 0})
    return phenotype

# get anatomical by id
def get_anatomical_by_id(full_id):
    anatomical = ANATOMICAL_COLLECTION.find_one({"id": full_id}, {'_id': 0})
    return anatomical

# get relationship by id
def get_relationship_by_id(full_id):
    relationship = RO_COLLECTION.find_one({"id": full_id}, {'_id': 0})
    return relationship

def save_data_model(data_model):
    DATA_MODEL_COLLECTION.delete_many({})
    DATA_MODEL_COLLECTION.insert_one(data_model)
    

def save_disease_dict(disease_dict):
    DISEASES_COLLECTION.delete_many({})
    DISEASES_COLLECTION.insert_many(disease_dict.values())
    
def save_phenotypes(phenotype_dict):
    PHENOTYPES_COLLECTION.delete_many({})
    PHENOTYPES_COLLECTION.insert_many(phenotype_dict.values())

def save_anatomical(anatomical_dict):
    ANATOMICAL_COLLECTION.delete_many({})
    ANATOMICAL_COLLECTION.insert_many(anatomical_dict.values())

def save_ro_dict(ro_dict):
    RO_COLLECTION.delete_many({})
    RO_COLLECTION.insert_many(ro_dict.values())

def save(data_model, disease_dict, phenotype_dict, anatomical_dict, ro_dict):
    """
    Save the data model and disease dictionary into MongoDB.

    This function clears existing data in the 'diseases' and 'data_model' collections and 
    inserts the new data.

    Parameters:
    data_model (dict): The data model to be saved.
    disease_dict (dict): A dictionary of disease entries to be saved.
    phenotype_dict (dict): A dictionary of phenotype entries to be saved.
    anatomical_dict (dict): A dictionary of uberon entries to be saved.
    """
    save_data_model(data_model)
    save_disease_dict(disease_dict)
    save_phenotypes(phenotype_dict)
    save_anatomical(anatomical_dict)
    save_ro_dict(ro_dict)
    
def set_hpo_graph():
    
    # TODO: CI-CD generate .ttl 
    print('loading hpo ontology')
    owl_file_path = "datasets/hpo/hp-base.owl"
    ttl_file_path = "datasets/hpo/hp-base.ttl"

    # load owl, save ttl
    graph = Graph()
    graph.parse(owl_file_path, format='xml')
    graph.serialize(destination=ttl_file_path, format='turtle')  
    
    with open(ttl_file_path, 'r') as file:
        ttl_data = file.read()

    ttl_data_bytes = ttl_data.encode('utf-8')

    fs.put(ttl_data_bytes, filename="ro.ttl")

    print('finished parsing hpo ontology')

    print('setting hpo graph')

    #services.wait_for_file_in_mongo("ro.ttl")
    #file = fs.find_one({"filename": "ro.ttl"})
    #ttl_data = file.read().decode('utf-8')
    # Copiar todos los triples del grafo original al nuevo grafo
    #for triple in graph:
    #    HPO_KG.add(triple)
    
    print('len de hpo graph: ', len(graph))

def wait_for_file_in_mongo(filename, timeout=300):
    """
    Wait until a specified file is available in MongoDB's GridFS.
    Is useful to avoid race condition with mongoDB

    Parameters:
    filename (str): The name of the file to wait for.
    timeout (int): The maximum time to wait for the file, in seconds.

    Raises:
    TimeoutError: If the file is not available within the timeout period.
    """
    start_time = time.time()
    while not fs.exists({"filename": filename}):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout: {filename} not available in mongoDB after {timeout} seconds.")
        print(f"Waiting for {filename} to be available in MongoDB...")
        time.sleep(5)
    print(f"{filename} is available MongoDB.")