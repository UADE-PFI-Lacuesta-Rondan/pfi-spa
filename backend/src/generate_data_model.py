import utils
import models.random_forest as random_forest
import models.ontogpt as ontogpt

import constants

import repository

def create_mondo_entry(id, name="Unknown", description="No description available", tracker_item=None):
    entry = {
        "id": id,
        "name": name,
        "description": description,
        "title": "Titulo de la enfermedad",
        "causes": ["Causa de la enfermedad #1", "Causa de la enfermedad #2", "Causa de la enfermedad #3"],
        "treatments": [],
        "anatomical_structures": [],
        "phenotypes": [],
        "exposures": [],
        "chemicals": [],
        "age_onsets": [],
        "children": [],
        "parent": None
    }
    if tracker_item:
        entry["tracker_item"] = tracker_item
    return entry

def create_entry(id, name="Unknown", description="No description available"):
    entry = {
        "id": id,
        "name": name,
        "description": description,
    }
    return entry

def build_is_a_hierarchy(mondo_data):
    is_a_hierarchy = {}
    for edge in mondo_data['graphs'][0]['edges']:
        if edge['pred'] == 'is_a':
            if edge['obj'] not in is_a_hierarchy:
                is_a_hierarchy[edge['obj']] = []
            is_a_hierarchy[edge['obj']].append(edge['sub'])
    return is_a_hierarchy

def get_all_descendants(node, hierarchy):
    descendants = set()
    nodes_to_visit = [node]
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if current_node in hierarchy:
            children = hierarchy[current_node]
            descendants.update(children)
            nodes_to_visit.extend(children)
    return descendants

def process_nodes(mondo_data, disease_dict, phenotype_dict, anatomical_dict, ro_dict, ecto_dict, maxo_dict, chebi_dict):

    # iterate nodes, get diseases (MONDO ID terms)
    for node in mondo_data['graphs'][0]['nodes']:

        node_id = node.get('id')

        # name and desc placeholders
        name = node.get('lbl', "Unknown Entity")
        description = "No description available"
        if 'meta' in node and 'definition' in node['meta']:
            description = node['meta']['definition'].get('val', description)

        # Excluir entidades obsoletas o de animales no humanos
        if "obsolete" in name.lower() or "non-human animal" in name.lower():
            continue

        # excluded entities, for example, top hierarchy nodes (MONDO:00000001 is disease, parent of them all)
        if node_id in constants.OMIT_ENTITIES:
            continue

        # phenotypes
        if "HP_" in node_id:
            phenotype_entry = create_entry(node_id, name, description)
            phenotype_dict[node_id] = phenotype_entry
            continue

        # anatomical
        if "UBERON_" in node_id:
            anatomical_entry = create_entry(node_id, name, description)
            anatomical_dict[node_id] = anatomical_entry
            continue

        # Relations Ontology (RO)
        if "RO_" in node_id:
            ro_entry = create_entry(node_id, name, description)
            ro_dict[node_id] = ro_entry
            continue

        # ECTO Ontology
        if "ECTO_" in node_id:
            ecto_entry = create_entry(node_id, name, description)
            ecto_dict[node_id] = ecto_entry
            continue

        # MAXO Ontology
        if "MAXO_" in node_id:
            entry = create_entry(node_id, name, description)
            maxo_dict[node_id] = entry
            continue

        # CHEBI Ontology
        if "CHEBI_" in node_id:
            entry = create_entry(node_id, name, description)
            chebi_dict[node_id] = entry
            continue

        # excluded triples
        # TODO: use regex
        if not node_id or constants.MONDO_STR not in node_id:
            continue

        # tracker item is the pull request where the disease has been added
        tracker_item = None
        if 'meta' in node and 'basicPropertyValues' in node['meta']:
            for bpv in node['meta']['basicPropertyValues']:
                if bpv['pred'] == constants.TRACK_ITEM_REL_TYPE:
                    tracker_item = bpv['val']
                    break

        if node_id not in disease_dict:
            disease_entry = create_mondo_entry(node_id, name, description, tracker_item)
            disease_dict[node_id] = disease_entry
        else:
            disease_entry = disease_dict[node_id]
            disease_entry['name'] = name
            disease_entry['description'] = description
            if tracker_item:
                disease_entry['tracker_item'] = tracker_item

def process_edges(mondo_data, age_onset_hierarchy, disease_dict, data_model, ro_dict, ecto_dict, maxo_dict, chebi_dict):

    # TODO refactor node labels
    node_labels = {node['id']: node.get('lbl', 'Unknown') for node in mondo_data['graphs'][0]['nodes']}

    relationships_types = {}

    # iterate triples
    for triple in mondo_data['graphs'][0]['edges']:

        subject_id = triple.get('sub')
        property_id = triple.get('pred')
        object_id = triple.get('obj')

        # for children and parent fields, use is_a relationship
        if property_id == constants.IS_A_RELATIONSHIP and subject_id in disease_dict and object_id in disease_dict:
            disease_dict[subject_id]['parent'] = object_id
            if 'children' not in disease_dict[object_id]:
                disease_dict[object_id]['children'] = []
            disease_dict[object_id]['children'].append(subject_id)

        # TODO check this limitation
        if property_id in constants.SUB_OF_PROPERTIES or object_id in constants.OMIT_ENTITIES:
            continue

        # only process relationships whose subject entity contains "MONDO" in its ID 
        if subject_id and property_id and object_id and constants.MONDO_STR in subject_id:
            if subject_id in disease_dict:
                disease_entry = disease_dict[subject_id]
                relationship_type = "has_relationship"
                object_label = node_labels.get(object_id, 'Unknown')

                relationship_entry = utils.create_relationship_entry(relationship_type, property_id, object_id, object_label)

                # TODO axel include GO relationships, discover new features to expand the model
                # TODO axel, seggregate the collection in different documents. 

                # Medical Actions Ontology
                if constants.MAXO_STR in object_id:
                    if relationship_entry not in disease_entry["treatments"]:
                        disease_entry["treatments"].append(relationship_entry)
                        if object_id not in data_model["treatment_to_diseases"]:
                            data_model["treatment_to_diseases"][object_id] = []
                        data_model["treatment_to_diseases"][object_id].append(subject_id)
                    
                    new_rel = {"type": constants.MAXO_STR, "target": property_id, "label": ""}
                    if property_id not in relationships_types:
                        relationships_types[property_id] = []
                    if new_rel not in relationships_types[property_id]:
                        relationships_types[property_id].append(new_rel)

                # Anatomical Parts
                elif constants.UBERON_STR in object_id:
                    if relationship_entry not in disease_entry["anatomical_structures"]:
                        disease_entry["anatomical_structures"].append(relationship_entry)
                        if object_id not in data_model["anatomical_to_diseases"]:
                            data_model["anatomical_to_diseases"][object_id] = []
                        data_model["anatomical_to_diseases"][object_id].append(subject_id)
                    
                    new_rel = {"type": constants.UBERON_STR, "target": property_id, "label": ""}
                    if property_id not in relationships_types:
                        relationships_types[property_id] = []
                    if new_rel not in relationships_types[property_id]:
                        relationships_types[property_id].append(new_rel)

                # Age Onset
                elif constants.HP_STR in object_id and object_id in age_onset_hierarchy:
                    if relationship_entry not in disease_entry["age_onsets"]:
                        disease_entry["age_onsets"].append(relationship_entry)
                        if object_id not in data_model["age_onset_to_diseases"]:
                            data_model["age_onset_to_diseases"][object_id] = []
                        data_model["age_onset_to_diseases"][object_id].append(subject_id)
                    
                    new_rel = {"type": constants.HP_STR, "target": property_id, "label": ""}
                    if property_id not in relationships_types:
                        relationships_types[property_id] = []
                    if new_rel not in relationships_types[property_id]:
                        relationships_types[property_id].append(new_rel)


                # Penotypes
                elif constants.HP_STR in object_id:
                    if relationship_entry not in disease_entry["phenotypes"]:
                        disease_entry["phenotypes"].append(relationship_entry)
                        if object_id not in data_model["phenotype_to_diseases"]:
                            data_model["phenotype_to_diseases"][object_id] = []
                        data_model["phenotype_to_diseases"][object_id].append(subject_id)

                    new_rel = {"type": constants.HP_STR, "target": property_id, "label": ""}
                    if property_id not in relationships_types:
                        relationships_types[property_id] = []
                    if new_rel not in relationships_types[property_id]:
                        relationships_types[property_id].append(new_rel)


                # Environmental Exposures, Conditions, Treatments 
                elif constants.ECTO_STR in object_id:
                    if relationship_entry not in disease_entry["exposures"]:
                        disease_entry["exposures"].append(relationship_entry)
                        if object_id not in data_model["exposure_to_diseases"]:
                            data_model["exposure_to_diseases"][object_id] = []
                        data_model["exposure_to_diseases"][object_id].append(subject_id)
                    
                    new_rel = {"type": constants.ECTO_STR, "target": property_id, "label": ""}
                    if property_id not in relationships_types:
                        relationships_types[property_id] = []
                    if new_rel not in relationships_types[property_id]:
                        relationships_types[property_id].append(new_rel)

                # Chemicals
                elif constants.CHEBI_STR in object_id:
                    if relationship_entry not in disease_entry["chemicals"]:
                        disease_entry["chemicals"].append(relationship_entry)
                        if object_id not in data_model["chemical_to_diseases"]:
                            data_model["chemical_to_diseases"][object_id] = []
                        data_model["chemical_to_diseases"][object_id].append(subject_id)
                    
                    new_rel = {"type": constants.CHEBI_STR, "target": property_id, "label": ""}
                    if property_id not in relationships_types:
                        relationships_types[property_id] = []
                    if new_rel not in relationships_types[property_id]:
                        relationships_types[property_id].append(new_rel)

                # TODO rest of relationships, GO
                #else:
                    # Only add non-MAXO, non-UBERON, and non-HP relationships to the relationships field
                 #   if relationship_entry not in disease_entry['relationships']:
                  #      disease_entry['relationships'].append(relationship_entry)
    
    data_model["relationships_types"] = relationships_types

def main():
    mondo_data = utils.load_json('datasets/mondo/mondo.json')

    age_onset_hierarchy = {
        constants.AGE_ONSET_PARENT_REL_TYPE: "Onset"
    }

    data_model = {
        "phenotype_to_diseases": {},
        "age_onset_to_diseases": {},
        "anatomical_to_diseases": {},
        "treatment_to_diseases": {},
        "exposure_to_diseases": {},
        "chemical_to_diseases": {},
        "relationships_types": {}
    }

    disease_dict = {}
    phenotypes_dict = {}
    anatomical_dict = {}
    ro_dict = {}
    ecto_dict = {}
    maxo_dict = {}
    chebi_dict = {}

    # TODO refactor hierarchy generated data
    # Build hierarchies
    is_a_hierarchy = build_is_a_hierarchy(mondo_data)
    onset_descendants = get_all_descendants(constants.AGE_ONSET_PARENT_REL_TYPE, is_a_hierarchy)
    for desc in onset_descendants:
        age_onset_hierarchy[desc] = "Onset"

    # Process data
    process_nodes(mondo_data, disease_dict, phenotypes_dict, anatomical_dict, ro_dict, ecto_dict, maxo_dict, chebi_dict) # TODO exclude obsolete terms from disease_dict
    process_edges(mondo_data, age_onset_hierarchy, disease_dict, data_model, ro_dict, ecto_dict, maxo_dict, chebi_dict)

    # Save to MongoDB
    repository.save(data_model, disease_dict, phenotypes_dict, anatomical_dict, ro_dict, ecto_dict, maxo_dict, chebi_dict)

    # hpo ontology collection
    #repository.set_hpo_graph()

    # apply train models

    #dbscan.clustering_data_frame()
    random_forest.generate_model()
    ontogpt.generate_model()


if __name__ == "__main__":
    main()
