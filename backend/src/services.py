import json
import os
import random
import tempfile

from pymongo import UpdateOne
import utils
import constants
import repository
import openai
import joblib


def set_llm_fields(disease):
    """
    Set on-the-fly generated fields by querying an LLM (Language Model). 
    Title, description, and causes fields for a given disease dictionary.

    Parameters:
    disease (dict): A dictionary containing information about the disease. The 'name' key must be present.

    This function constructs a prompt to query the OpenAI API to traverse the MONDO disease ontology and 
    retrieve the title, description, and causes of the disease. The retrieved information is then added 
    to the input dictionary.

    The format of the prompt ensures that the response is in JSON format, with specific fields: title, 
    description, and causes.

    Raises:
    KeyError: If 'name' is not present in the disease dictionary.
    """
    text = "This is the name of a disease in MONDO disease ontology: " + disease['name']
    text = text + ". Please traverse the ontology and get a title, description and causes of the disease. Do not include any additional text as the output, it has to have the following format, in JSON. Exclude json decorations, only keep json structure: title: title of the disease, description: brief description of the disease, causes: List of brief causes."
           
     # Initialize the OpenAI client with your API key
    client = None
    apiKey = os.getenv('OPENAI_API_KEY', '')
    if len(apiKey) > 0: 
        client = openai.OpenAI(api_key=apiKey)
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Mondo Ontology traverse assistant."},
            {"role": "user", "content": text}
        ])

        result_content = completion.choices[0].message.content

        # Convertir result_content de cadena JSON a diccionario de Python
        result_json = json.loads(result_content)

        disease['description'] = result_json['description']
        disease['title'] = result_json['title']
        disease['causes'] = result_json['causes']

def predict_relationship(disease_id, relationship_type, relationship_property, model_name):
    """
    Predict the target ID for a given disease ID, relationship type, and relationship property
    using a pre-trained Random Forest model stored in MongoDB.

    Parameters:
    disease_id (str): The identifier for the disease.
    relationship_type (str): The type of the relationship.
    relationship_property (str): The property of the relationship.

    Returns:
    str: The predicted target ID for the given inputs.
    """
    
    # random forest files stored in fileGrid
    RANDOM_FOREST_MODEL_FILES = [
        f'{model_name}.pkl',
        f'le_disease_{model_name}.pkl',
        f'le_relationship_property_{model_name}.pkl',
        f'le_target_id_{model_name}.pkl',
        f'le_disease_rel_prop_{model_name}.pkl',
        f'le_cluster_{model_name}.pkl',
    ]

    # wait if necessary
    for filename in RANDOM_FOREST_MODEL_FILES:
        repository.wait_for_file_in_mongo(filename)

    model = load_json_from_mongo(f'{model_name}.pkl')
    le_disease = load_json_from_mongo(f'le_disease_{model_name}.pkl')
    le_relationship_property = load_json_from_mongo(f'le_relationship_property_{model_name}.pkl')
    le_target_id = load_json_from_mongo(f'le_target_id_{model_name}.pkl')
    le_disease_rel_prop = load_json_from_mongo(f'le_disease_rel_prop_{model_name}.pkl')
    le_cluster = load_json_from_mongo(f'le_cluster_{model_name}.pkl')

    # encode inputs
    disease_id_encoded = le_disease.transform([disease_id])[0]
    relationship_property_encoded = le_relationship_property.transform([relationship_property])[0]

    # feature engineering transcode of the filters
    disease_rel_prop = f"{disease_id_encoded}_{relationship_property_encoded}"

    # not seen labels handling
    if disease_rel_prop not in le_disease_rel_prop.classes_:
        disease_rel_prop_encoded = -1
    else:
        disease_rel_prop_encoded = le_disease_rel_prop.transform([disease_rel_prop])[0]

    # cluster feature
    cluster_feature = f"{disease_id_encoded}_{relationship_property_encoded}"
    if cluster_feature not in le_cluster.classes_:
        cluster_encoded = -1  # Handle unseen cluster labels
    else:
        cluster_encoded = le_cluster.transform([cluster_feature])[0]

    input_features = [
        disease_id_encoded,
        relationship_property_encoded,
        disease_rel_prop_encoded,
        cluster_encoded
    ]

    # predict
    target_encoded = model.predict([input_features])
    le_target_id.inverse_transform(target_encoded)

    # posible targets
    possible_targets = [t for t in le_target_id.classes_ if is_valid_relationship(relationship_property, t)]

    # randomize to not be deterministic
    predicted_target = random.choice(possible_targets)

    return predicted_target

def is_valid_relationship(property_id, target_id):
    """
    Check if the relationship between a given property ID and target ID is valid based on the 
    predefined types of relationships.

    Parameters:
    property_id (str): The identifier for the property (relationship type).
    target_id (str): The identifier for the target entity.

    Returns:
    bool: True if the relationship is valid, False otherwise.
    """

    relationships_types = repository.get_data_model()['relationships_types']
    if property_id in relationships_types:
        valid_types = relationships_types[property_id]
        for relationship in valid_types:
            if relationship["type"] == constants.UBERON_STR and constants.UBERON_STR in target_id:
                return True
            if relationship["type"] == constants.HP_STR and constants.HP_STR in target_id:
                return True
            if relationship["type"] == constants.ECTO_STR and constants.ECTO_STR in target_id:
                return True
            if relationship["type"] == constants.MAXO_STR and constants.MAXO_STR in target_id:
                return True
            if relationship["type"] == constants.CHEBI_STR and constants.CHEBI_STR in target_id:
                return True
        return False
    return True

def get_relationship_type_for_property(property_id):
    relationships_types = repository.get_data_model()['relationships_types']
    if property_id in relationships_types:
        valid_types = relationships_types[property_id]
        for relationship in valid_types:
            if relationship["target"] == property_id:
                return relationship["type"]
    return None

def get_hierarchy_by_mondo_id(mondo_id):

    """
    Generate a hierarchical representation of diseases starting from a given MONDO ID.

    This function retrieves the hierarchy of diseases starting from a specified MONDO ID
    and represents it in a format suitable for visualization. It builds a tree structure
    where each node represents a disease and its relationship to its parent.

    Parameters:
    mondo_id (str): The MONDO ID of the root disease to start the hierarchy from.

    Returns:
    list: A list of lists representing the hierarchy. Each sublist contains information
          about a disease node in the format [id, childLabel, parent, size, {"role": "style"}, "link"].

    The function performs the following steps:
    1. Fetches all diseases from the MongoDB collection and constructs a dictionary for quick lookup.
    2. Initializes the hierarchy list with the headers.
    3. Defines helper functions to add nodes to the hierarchy and to process the hierarchy recursively.
    4. Builds the hierarchy starting from the given MONDO ID.
    """

    # TODO improve
    diseases = list(repository.DISEASES_COLLECTION.find({}, {'_id': 0}))
    disease_dict = {d['id']: d for d in diseases}

    # TODO include link
    #hierarchy = [["id", "childLabel", "parent", "size", {"role": "style"}, "link"]]
    hierarchy = [["id", "childLabel", "parent", "size", {"role": "style"}]]
    id_map = {}
    current_id = 0

    def add_to_hierarchy(disease_id, parent_id, label, size, color):
        nonlocal current_id
        id_map[disease_id] = current_id
        #hierarchy.append([current_id, label, parent_id, size, color, disease_id])
        hierarchy.append([current_id, label, parent_id, size, color])
        current_id += 1

    def process_hierarchy(disease_id, parent_id):
        try:
            disease = disease_dict[disease_id]
        except KeyError:
            return
        
        add_to_hierarchy(disease_id, parent_id, disease['name'], 1, "#1d8bf8")
        for child_id in disease.get('children', []):
            process_hierarchy(child_id, id_map[disease_id])

    if mondo_id in disease_dict:
        add_to_hierarchy(mondo_id, -1, disease_dict[mondo_id]['name'], 1, "#1d8bf8")
        for child_id in disease_dict[mondo_id].get('children', []):
            process_hierarchy(child_id, id_map[mondo_id])

    return hierarchy

def get_extended_hierarchy_by_mondo_id(mondo_id):
    """
    Generate a hierarchical representation of diseases, phenotypes, and anatomical structures starting from a given MONDO ID.
    This function retrieves the hierarchy starting from a specified MONDO ID and includes branches for phenotypes and anatomical structures.
    
    Parameters:
    mondo_id (str): The MONDO ID of the root disease to start the hierarchy from.

    Returns:
    tuple: A tuple containing the hierarchy list and a legend dict.
    """

    # TODO improve
    diseases = list(repository.DISEASES_COLLECTION.find({}, {'_id': 0}))
    disease_dict = {d['id']: d for d in diseases}

    # TODO link
    #hierarchy = [["id", "childLabel", "parent", "size", {"role": "style"}, "link"]]
    hierarchy = [["id", "childLabel", "parent", "size", {"role": "style"}]]
    id_map = {}
    current_id = 0

    legend = {
        "Disease": "#0a0a0a",
        "Phenotype": "#1d522a",
        "Anatomical Structure": "#1d3952",
        "Exposure": "#1d4022",
        "Treatment": "#2d4022",
        "Chemical": "#3d4022",
        "Predicted": "#e5ff00"
    }

    def add_to_hierarchy(node_id, parent_id, label, size, color):
        nonlocal current_id
        id_map[node_id] = current_id
        #hierarchy.append([current_id, label, parent_id, size, color, node_id])
        hierarchy.append([current_id, label, parent_id, size, color])
        current_id += 1

    def process_hierarchy(disease_id, parent_id):
        try:
            disease = disease_dict[disease_id]
        except KeyError:
            return
        add_to_hierarchy(disease_id, parent_id, disease['name'], 1, legend["Disease"])
        
        # Process children diseases
        for child_id in disease.get('children', []):
            process_hierarchy(child_id, id_map[disease_id])
        
        # Process phenotypes
        for phenotype in disease.get('phenotypes', []):
            phenotype_id = phenotype['target']
            color = legend["Predicted"] if phenotype.get('predicted', False) else legend["Phenotype"]
            add_to_hierarchy(phenotype_id, id_map[disease_id], phenotype['label'], 1, color)
        
        # Process anatomical structures
        for anatomical in disease.get('anatomical_structures', []):
            anatomical_id = anatomical['target']
            color = legend["Predicted"] if anatomical.get('predicted', False) else legend["Anatomical Structure"]
            add_to_hierarchy(anatomical_id, id_map[disease_id], anatomical['label'], 1, color)

        # Process Exposures
        for exposure in disease.get('exposures', []):
            exposure_id = exposure['target']
            color = legend["Predicted"] if exposure.get('predicted', False) else legend["Exposure"]
            add_to_hierarchy(exposure_id, id_map[disease_id], exposure['label'], 1, color)
        
        # Process Treatments
        for treatment in disease.get('treatments', []):
            treatment_id = treatment['target']
            color = legend["Predicted"] if treatment.get('predicted', False) else legend["Treatment"]
            add_to_hierarchy(treatment_id, id_map[disease_id], treatment['label'], 1, color)

        # Process Chemicals
        for chemical in disease.get('chemicals', []):
            chemical_id = chemical['target']
            color = legend["Predicted"] if chemical.get('predicted', False) else legend["Chemical"]
            add_to_hierarchy(chemical_id, id_map[disease_id], chemical['label'], 1, color)

    if mondo_id in disease_dict:
        add_to_hierarchy(mondo_id, -1, disease_dict[mondo_id]['name'], 1, legend["Disease"])
        process_hierarchy(mondo_id, id_map[mondo_id])

    return hierarchy, legend

def get_diseases_by_phenotypes(phenotype_ids):
    """
    Retrieve all diseases associated with all of the given phenotypes, excluding the specified phenotypes from the phenotypes list.

    :param phenotype_ids: A list of phenotype IDs (HP IDs).

    :return: A list of diseases associated with all the given phenotypes.
    """
    if not phenotype_ids:
        return []
    
    # retrieve all disease IDs from the DISEASES collection
    all_disease_ids = set(repository.get_diseases_ids())

    if not all_disease_ids:
        return []
    
    filtered_disease_ids = all_disease_ids
    data_model = repository.get_data_model()

    print(f"Filtering diseases by phenotypes: {phenotype_ids}")    

    initial_phenotype_id = phenotype_ids[0]
    phenotype_disease_ids = set(data_model['phenotype_to_diseases'].get(initial_phenotype_id, []))
    for phenotype_id in phenotype_ids[1:]:
        current_disease_ids = set(data_model['phenotype_to_diseases'].get(phenotype_id, []))
        phenotype_disease_ids.intersection_update(current_disease_ids)
    filtered_disease_ids.intersection_update(phenotype_disease_ids)

    # diseases that match all filtered disease ids
    diseases = repository.get_diseases_by_ids(filtered_disease_ids)

    print(f"Matching diseases: {diseases}")

    return diseases

def get_diseases_by_age_onsets(age_onset_ids):
    """
    Retrieve all diseases associated with any of the given age onsets, excluding the specified age onsets from the age_onsets list.

    :param age_onset_ids: A list of age onset IDs (HP IDs).
    :param data_model: The data model containing diseases and their relationships.
    :return: A list of diseases associated with the given age onsets.
    """
    if not age_onset_ids:
        return []

    all_disease_ids = set(repository.get_diseases_ids())

    if not all_disease_ids:
        return []

    filtered_disease_ids = all_disease_ids
    data_model = repository.get_data_model()

    initial_age_onset_id = age_onset_ids[0]
    age_onset_disease_ids = set(data_model['age_onset_to_diseases'].get(initial_age_onset_id, []))
    for age_onset_id in age_onset_ids[1:]:
        current_disease_ids = set(data_model['age_onset_to_diseases'].get(age_onset_id, []))
        age_onset_disease_ids.intersection_update(current_disease_ids)
    filtered_disease_ids.intersection_update(age_onset_disease_ids)

    diseases = repository.get_diseases_by_ids(filtered_disease_ids)

    return diseases

def get_diseases_by_anatomical_structures(anatomical_ids):
    """
    Retrieve all diseases associated with any of the given anatomical structures, excluding the specified anatomical structures from the anatomical_structures list.

    :param anatomical_ids: A list of anatomical structure IDs (UBERON IDs).
    :param data_model: The data model containing diseases and their relationships.
    :return: A list of diseases associated with the given anatomical structures.
    """
    if not anatomical_ids:
        return []

    all_disease_ids = set(repository.get_diseases_ids())

    if not all_disease_ids:
        return []

    filtered_disease_ids = all_disease_ids
    data_model = repository.get_data_model()

    initial_anatomical_id = anatomical_ids[0]
    anatomical_disease_ids = set(data_model['anatomical_to_diseases'].get(initial_anatomical_id, []))
    for anatomical_id in anatomical_ids[1:]:
        current_disease_ids = set(data_model['anatomical_to_diseases'].get(anatomical_id, []))
        anatomical_disease_ids.intersection_update(current_disease_ids)
    filtered_disease_ids.intersection_update(anatomical_disease_ids)

    diseases = repository.get_diseases_by_ids(filtered_disease_ids)

    return diseases

def get_diseases_by_exposures(exposure_ids):
    """
    Retrieve all diseases associated with all of the given exposures, excluding the specified exposures from the exposures list.

    :param exposure_ids: A list of exposure IDs (ECTO IDs).

    :return: A list of diseases associated with all the given exposures.
    """
    if not exposure_ids:
        return []
    
    # retrieve all disease IDs from the DISEASES collection
    all_disease_ids = set(repository.get_diseases_ids())

    if not all_disease_ids:
        return []
    
    filtered_disease_ids = all_disease_ids
    data_model = repository.get_data_model()

    print(f"Filtering diseases by exposures: {exposure_ids}")    

    initial_exposure_id = exposure_ids[0]
    exposure_disease_ids = set(data_model['exposure_to_diseases'].get(initial_exposure_id, []))
    for exposure_id in exposure_ids[1:]:
        current_disease_ids = set(data_model['exposure_to_diseases'].get(exposure_id, []))
        exposure_disease_ids.intersection_update(current_disease_ids)
    filtered_disease_ids.intersection_update(exposure_disease_ids)

    # diseases that match all filtered disease ids
    diseases = repository.get_diseases_by_ids(filtered_disease_ids)

    print(f"Matching diseases: {diseases}")

    return diseases


def get_diseases_by_treatments(treatment_ids):
    """
    Retrieve all diseases associated with all of the given treatments, excluding the specified treatments from the treatments list.

    :param treatment_ids: A list of treatment IDs (MAXO IDs).

    :return: A list of diseases associated with all the given treatments.
    """
    if not treatment_ids:
        return []
    
    # retrieve all disease IDs from the DISEASES collection
    all_disease_ids = set(repository.get_diseases_ids())

    if not all_disease_ids:
        return []
    
    filtered_disease_ids = all_disease_ids
    data_model = repository.get_data_model()

    print(f"Filtering diseases by treatments: {treatment_ids}")    

    initial_treatment_id = treatment_ids[0]
    treatment_disease_ids = set(data_model['treatment_to_diseases'].get(initial_treatment_id, []))
    for treatment_id in treatment_ids[1:]:
        current_disease_ids = set(data_model['treatment_to_diseases'].get(treatment_id, []))
        treatment_disease_ids.intersection_update(current_disease_ids)
    filtered_disease_ids.intersection_update(treatment_disease_ids)

    # diseases that match all filtered disease ids
    diseases = repository.get_diseases_by_ids(filtered_disease_ids)

    print(f"Matching diseases: {diseases}")

    return diseases


def get_diseases_by_chemicals(chemical_ids):
    """
    Retrieve all diseases associated with all of the given chemicals, excluding the specified chemicals from the chemicals list.

    :param chemical_ids: A list of chemical IDs (CHEBI IDs).

    :return: A list of diseases associated with all the given chemicals.
    """
    if not chemical_ids:
        return []
    
    # retrieve all disease IDs from the DISEASES collection
    all_disease_ids = set(repository.get_diseases_ids())

    if not all_disease_ids:
        return []
    
    filtered_disease_ids = all_disease_ids
    data_model = repository.get_data_model()

    print(f"Filtering diseases by chemicals: {chemical_ids}")    

    initial_chemical_id = chemical_ids[0]
    chemical_disease_ids = set(data_model['chemical_to_diseases'].get(initial_chemical_id, []))
    for chemical_id in chemical_ids[1:]:
        current_disease_ids = set(data_model['chemical_to_diseases'].get(chemical_id, []))
        chemical_disease_ids.intersection_update(current_disease_ids)
    filtered_disease_ids.intersection_update(chemical_disease_ids)

    # diseases that match all filtered disease ids
    diseases = repository.get_diseases_by_ids(filtered_disease_ids)

    print(f"Matching diseases: {diseases}")

    return diseases

def get_diseases_by_filters(phenotype_ids, anatomical_ids, age_onset_ids, exposure_ids, treatment_ids, chemical_ids):
    """
    Retrieve all diseases associated with all of the given filters

    :return: A list of diseases associated with all the given filters.
    """
    
    data_model = repository.get_data_model()

    if not (phenotype_ids or anatomical_ids or age_onset_ids or exposure_ids  or treatment_ids or chemical_ids):
      return []

    all_disease_ids = set(repository.get_diseases_ids())
    if not all_disease_ids:
        return []
    
    # filter disease IDs based on the provided filters
    filtered_disease_ids = all_disease_ids

    # filtering logic
    if phenotype_ids:
        initial_phenotype_id = phenotype_ids[0]
        phenotype_disease_ids = set(data_model['phenotype_to_diseases'].get(initial_phenotype_id, []))
        for phenotype_id in phenotype_ids[1:]:
            current_disease_ids = set(data_model['phenotype_to_diseases'].get(phenotype_id, []))
            phenotype_disease_ids.intersection_update(current_disease_ids)
        filtered_disease_ids.intersection_update(phenotype_disease_ids)

    if anatomical_ids:
        initial_anatomical_id = anatomical_ids[0]
        anatomical_disease_ids = set(data_model['anatomical_to_diseases'].get(initial_anatomical_id, []))
        for anatomical_id in anatomical_ids[1:]:
            current_disease_ids = set(data_model['anatomical_to_diseases'].get(anatomical_id, []))
            anatomical_disease_ids.intersection_update(current_disease_ids)
        filtered_disease_ids.intersection_update(anatomical_disease_ids)

    if age_onset_ids:
        initial_age_onset_id = age_onset_ids[0]
        age_onset_disease_ids = set(data_model['age_onset_to_diseases'].get(initial_age_onset_id, []))
        for age_onset_id in age_onset_ids[1:]:
            current_disease_ids = set(data_model['age_onset_to_diseases'].get(age_onset_id, []))
            age_onset_disease_ids.intersection_update(current_disease_ids)
        filtered_disease_ids.intersection_update(age_onset_disease_ids)

    if exposure_ids:
        initial_exposure_id = exposure_ids[0]
        exposure_disease_ids = set(data_model['exposure_to_diseases'].get(initial_exposure_id, []))
        for exposure_id in exposure_ids[1:]:
            current_disease_ids = set(data_model['exposure_to_diseases'].get(exposure_id, []))
            exposure_disease_ids.intersection_update(current_disease_ids)
        filtered_disease_ids.intersection_update(exposure_disease_ids)

    if treatment_ids:
        initial_treatment_id = treatment_ids[0]
        treatment_disease_ids = set(data_model['treatment_to_diseases'].get(initial_treatment_id, []))
        for treatment_id in treatment_ids[1:]:
            current_disease_ids = set(data_model['treatment_to_diseases'].get(treatment_id, []))
            treatment_disease_ids.intersection_update(current_disease_ids)
        filtered_disease_ids.intersection_update(treatment_disease_ids)

    if chemical_ids:
        initial_chemical_id = chemical_ids[0]
        chemical_disease_ids = set(data_model['chemical_to_diseases'].get(initial_chemical_id, []))
        for chemical_id in chemical_ids[1:]:
            current_disease_ids = set(data_model['chemical_to_diseases'].get(chemical_id, []))
            chemical_disease_ids.intersection_update(current_disease_ids)
        filtered_disease_ids.intersection_update(chemical_disease_ids)

    # retrieve the diseases that match all filtered disease ids
    return repository.get_diseases_by_ids(filtered_disease_ids)

# load model files
def load_json_from_mongo(filename):
    with tempfile.NamedTemporaryFile() as temp_file:
        with repository.fs.get_last_version(filename) as file_data:
            temp_file.write(file_data.read())
            temp_file.flush()
            return joblib.load(temp_file.name)
        
def get_phenotypes():
    """
    Retrieve a list of unique phenotypes from the data model.
    Returns: a list of dictionaries representing unique phenotypes. Each dictionary contains:
          - The label of the phenotype.
          - The identifier of the phenotype, with the URL prefix removed.
    """

    phenotypes = []
    # TODO, read from data_model phenotypes to diseases, then match to its collection to get label
    for disease in repository.get_diseases():
        for phenotype in disease.get('phenotypes', []):
            phenotypes.append({
                "label": phenotype.get('label', 'Unknown Phenotype'),
                "value": phenotype['target'].replace("http://purl.obolibrary.org/obo/", "")
            })
    
    # Remove duplicates
    unique_phenotypes = {v['value']: v for v in phenotypes}.values()
    return list(unique_phenotypes)

def get_anatomical_structures():
    """
    Retrieve a list of unique anatomical_structure from the data model.
    Returns: a list of dictionaries representing unique anatomical_structures. Each dictionary contains:
          - The label of the anatomical_structure.
          - The identifier of the anatomical_structure, with the URL prefix removed.
    """

    anatomical_structures = []
    # TODO, read from data_model anatomical_structure to diseases, then match to its collection to get label
    for disease in repository.get_diseases():
        for anatomical_structure in disease.get('anatomical_structures', []):
            anatomical_structures.append({
                "label": anatomical_structure.get('label', 'Unknown anatomical structure'),
                "value": anatomical_structure['target'].replace("http://purl.obolibrary.org/obo/", "")
            })
    
    # Remove duplicates
    unique_anatomical_structures = {v['value']: v for v in anatomical_structures}.values()
    return list(unique_anatomical_structures)

def get_age_onsets():
    """
    Retrieve a list of unique age_onsets from the data model.
    Returns: a list of dictionaries representing unique age_onsets. Each dictionary contains:
          - The label of the age_onsets.
          - The identifier of the age_onsets, with the URL prefix removed.
    """
    age_onsets = []
    # TODO, read from data_model ageonset to diseases, then match to its collection to get label
    for disease in repository.get_diseases():
        for age_onset in disease.get('age_onsets', []):
            age_onsets.append({
                "label": age_onset.get('label', 'Unknown age onset'),
                "value": age_onset['target'].replace("http://purl.obolibrary.org/obo/", "")
            })

    # Remove duplicates
    unique_age_onsets = {v['value']: v for v in age_onsets}.values()
    return list(unique_age_onsets)


def get_exposures():
    """
    Retrieve a list of unique exposure from the data model.
    Returns: a list of dictionaries representing unique exposures. Each dictionary contains:
          - The label of the exposure.
          - The identifier of the exposure, with the URL prefix removed.
    """

    exposures = []
    # TODO, read from data_model exposure to diseases, then match to its collection to get label
    for disease in repository.get_diseases():
        for exposure in disease.get('exposures', []):
            exposures.append({
                "label": exposure.get('label', 'Unknown exposure'),
                "value": exposure['target'].replace("http://purl.obolibrary.org/obo/", "")
            })
    
    # Remove duplicates
    unique_exposures = {v['value']: v for v in exposures}.values()
    return list(unique_exposures)

def get_treatments():
    """
    Retrieve a list of unique treatment from the data model.
    Returns: a list of dictionaries representing unique treatments. Each dictionary contains:
          - The label of the treatment.
          - The identifier of the treatment, with the URL prefix removed.
    """

    treatments = []
    # TODO, read from data_model treatment to diseases, then match to its collection to get label
    for disease in repository.get_diseases():
        for treatment in disease.get('treatments', []):
            treatments.append({
                "label": treatment.get('label', 'Unknown treatment'),
                "value": treatment['target'].replace("http://purl.obolibrary.org/obo/", "")
            })
    
    # Remove duplicates
    unique_treatments = {v['value']: v for v in treatments}.values()
    return list(unique_treatments)

def get_chemicals():
    """
    Retrieve a list of unique chemical from the data model.
    Returns: a list of dictionaries representing unique chemicals. Each dictionary contains:
          - The label of the chemical.
          - The identifier of the chemical, with the URL prefix removed.
    """

    chemicals = []
    # TODO, read from data_model chemical to diseases, then match to its collection to get label
    for disease in repository.get_diseases():
        for chemical in disease.get('chemicals', []):
            chemicals.append({
                "label": chemical.get('label', 'Unknown chemical'),
                "value": chemical['target'].replace("http://purl.obolibrary.org/obo/", "")
            })
    
    # Remove duplicates
    unique_chemicals = {v['value']: v for v in chemicals}.values()
    return list(unique_chemicals)

# get phenotype by id
def get_phenotype_by_id(full_id):
    return repository.get_phenotype_by_id(full_id)

# get anatomical by id
def get_anatomical_by_id(full_id):
    return repository.get_anatomical_by_id(full_id)

# get anatomical by id
def get_relationship_by_id(full_id):
    return repository.get_relationship_by_id(full_id)

# get exposure by id
def get_exposure_by_id(full_id):
    return repository.get_exposure_by_id(full_id)

# get treatment by id
def get_treatment_by_id(full_id):
    return repository.get_treatment_by_id(full_id)

# get chemical by id
def get_chemical_by_id(full_id):
    return repository.get_chemical_by_id(full_id)
  
def get_relationship_types():
    relationship_types = []
    for key, _ in repository.get_data_model()['relationships_types'].items():

        ro_term = get_relationship_by_id(key)
        if ro_term is not None and "name" in ro_term:
            relationship_types.append({
                "label": ro_term["name"],
                "value": key
            })

    # Remove duplicates
    unique_relationship_types = {v['value']: v for v in relationship_types}.values()
    return list(unique_relationship_types)

def update_data_model(full_disease_id, full_new_relationship_property, predicted_target):
    
    phenotype = get_phenotype_by_id(predicted_target) if constants.HP_STR in predicted_target else None
    anatomical_structure = get_anatomical_by_id(predicted_target) if constants.UBERON_STR in predicted_target else None
    exposure = get_exposure_by_id(predicted_target) if constants.ECTO_STR in predicted_target else None
    treatment = get_treatment_by_id(predicted_target) if constants.MAXO_STR in predicted_target else None
    chemical = get_chemical_by_id(predicted_target) if constants.CHEBI_STR in predicted_target else None

    data_model = repository.get_data_model()
    relationships = data_model["relationships_types"].get(full_new_relationship_property, [])

    update_operations = []
    for relationship_type in relationships:
        relationship_type_str = relationship_type["type"]

        if relationship_type_str == constants.HP_STR:
            phenotype_to_diseases = data_model["phenotype_to_diseases"].get(predicted_target, [])
            if full_disease_id not in phenotype_to_diseases:
                phenotype_to_diseases.append(full_disease_id)
            data_model["phenotype_to_diseases"][predicted_target] = phenotype_to_diseases
        
        elif relationship_type_str == constants.UBERON_STR:
            anatomical_to_diseases = data_model["anatomical_to_diseases"].get(predicted_target, [])
            if full_disease_id not in anatomical_to_diseases:
                anatomical_to_diseases.append(full_disease_id)
            data_model["anatomical_to_diseases"][predicted_target] = anatomical_to_diseases
        
        elif relationship_type_str == constants.ECTO_STR:
            exposure_to_diseases = data_model["exposure_to_diseases"].get(predicted_target, [])
            if full_disease_id not in exposure_to_diseases:
                exposure_to_diseases.append(full_disease_id)
            data_model["exposure_to_diseases"][predicted_target] = exposure_to_diseases
        
        elif relationship_type_str == constants.MAXO_STR:
            treatment_to_diseases = data_model["treatment_to_diseases"].get(predicted_target, [])
            if full_disease_id not in treatment_to_diseases:
                treatment_to_diseases.append(full_disease_id)
            data_model["treatment_to_diseases"][predicted_target] = treatment_to_diseases
        
        elif relationship_type_str == constants.CHEBI_STR:
            chemical_to_diseases = data_model["chemical_to_diseases"].get(predicted_target, [])
            if full_disease_id not in chemical_to_diseases:
                chemical_to_diseases.append(full_disease_id)
            data_model["chemical_to_diseases"][predicted_target] = chemical_to_diseases

        for disease in repository.get_diseases():
            if disease["id"] == full_disease_id:
                update_fields = {}

                if relationship_type_str == constants.HP_STR and phenotype:
                    if "phenotypes" not in disease:
                        disease["phenotypes"] = []
                    disease["phenotypes"].append(utils.create_relationship_entry(
                        "has_relationship", 
                        full_new_relationship_property, 
                        predicted_target, 
                        phenotype["name"], True))
                    update_fields["phenotypes"] = disease["phenotypes"]
                
                elif relationship_type_str == constants.UBERON_STR and anatomical_structure:
                    if "anatomical_structures" not in disease:
                        disease["anatomical_structures"] = []
                    disease["anatomical_structures"].append(utils.create_relationship_entry(
                        "has_relationship", 
                        full_new_relationship_property, 
                        predicted_target, 
                        anatomical_structure["name"], True))
                    update_fields["anatomical_structures"] = disease["anatomical_structures"]

                elif relationship_type_str == constants.ECTO_STR and exposure:
                    if "exposures" not in disease:
                        disease["exposures"] = []
                    disease["exposures"].append(utils.create_relationship_entry(
                        "has_relationship", 
                        full_new_relationship_property, 
                        predicted_target, 
                        exposure["name"], True))
                    update_fields["exposures"] = disease["exposures"]

                elif relationship_type_str == constants.MAXO_STR and treatment:
                    if "treatments" not in disease:
                        disease["treatments"] = []
                    disease["treatments"].append(utils.create_relationship_entry(
                        "has_relationship", 
                        full_new_relationship_property, 
                        predicted_target, 
                        treatment["name"], True))
                    update_fields["treatments"] = disease["treatments"]

                elif relationship_type_str == constants.CHEBI_STR and chemical:
                    if "chemicals" not in disease:
                        disease["chemicals"] = []
                    disease["chemicals"].append(utils.create_relationship_entry(
                        "has_relationship", 
                        full_new_relationship_property, 
                        predicted_target, 
                        chemical["name"], True))
                    update_fields["chemicals"] = disease["chemicals"]
                
                if update_fields:
                    update_operations.append(
                        UpdateOne(
                            {"id": full_disease_id},
                            {"$set": update_fields}
                        )
                    )

    if update_operations:
        repository.DISEASES_COLLECTION.bulk_write(update_operations)
    repository.save_data_model(data_model)

def get_relationship_by_id_sparql(ro_id):

    sparql_query = constants.LABEL_QUERY.replace("RO_0000000", ro_id)
    graph = repository.get_hpo_kg()
    results = graph.query(sparql_query)

    result_list = []
    for row in results:
        result_list.append({str(var): str(row[var]) for var in row.labels})
    return result_list