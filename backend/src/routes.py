import json
import poplib
import tempfile
import constants
from flask import jsonify, make_response, request
import services
import repository
import utils

# TODO axel version endpoints
# TODO implement controller-services-repository pattern
def init_routes(app):

    @app.route("/disease/<mondo_id>", methods=["GET"])
    def get_disease(mondo_id):
        full_id = f"http://purl.obolibrary.org/obo/{mondo_id}"
        disease = repository.get_disease_by_id(full_id)
        if disease:
            try:
               services.set_llm_fields(disease)
            except Exception as e:
                return create_json_response(jsonify({"error": str(e)}), 500)

            return create_json_response(jsonify(disease), 200)
        else:
            return create_json_response(jsonify("Disease not found"), 404)

    @app.route("/filter_hierarchy/<mondo_id>", methods=["GET"])
    def filter_hierarchy(mondo_id):
        full_id = f"http://purl.obolibrary.org/obo/{mondo_id}"
        if not repository.get_disease_by_id(full_id):
            return create_json_response(jsonify("Disease not found"), 404)
        hierarchy = services.get_hierarchy_by_mondo_id(full_id)

        extended_hierarchy = services.get_extended_hierarchy_by_mondo_id(full_id)

        return create_json_response(jsonify({"hierarchy": hierarchy, "extended_hierarchy": extended_hierarchy}), 200)
    
    @app.route("/diseases/by_phenotypes", methods=["POST"])
    def diseases_by_phenotypes():
        phenotype_ids = request.json.get('phenotype_ids')
        if not phenotype_ids:
           return create_json_response(jsonify("Phenotype IDs are required"), 400)
        
        print(f"Received phenotype IDs: {phenotype_ids}")
        # Transform the phenotype IDs to include the full URI
        full_phenotype_ids = [f"http://purl.obolibrary.org/obo/{pid}" for pid in phenotype_ids]
        print(f"Transformed phenotype IDs: {full_phenotype_ids}")

        diseases = services.get_diseases_by_phenotypes(full_phenotype_ids)
        diseases = utils.convert_objectid_to_str(diseases)
        print(f"Found diseases: {diseases}")
        return create_json_response(jsonify(diseases), 200)
    
    @app.route("/diseases/by_age_onsets", methods=["POST"])
    def diseases_by_age_onsets():
        age_onset_ids = request.json.get('age_onset_ids')
        if not age_onset_ids:
           return create_json_response(jsonify("Age onset IDs are required"), 400)
        
        print(f"Received age onset IDs: {age_onset_ids}")
        # Transform the age onset IDs to include the full URI
        full_age_onset_ids = [f"http://purl.obolibrary.org/obo/{aid}" for aid in age_onset_ids]
        print(f"Transformed age onset IDs: {full_age_onset_ids}")

        diseases = services.get_diseases_by_age_onsets(full_age_onset_ids)
        diseases = utils.convert_objectid_to_str(diseases)
        print(f"Found diseases: {diseases}")
        return create_json_response(jsonify(diseases), 200)

    @app.route("/diseases/by_anatomical_structures", methods=["POST"])
    def diseases_by_anatomical_structures():
        anatomical_ids = request.json.get('anatomical_ids')
        if not anatomical_ids:
            return create_json_response(jsonify("Anatomical structure IDs are required"), 400)
        
        print(f"Received anatomical structure IDs: {anatomical_ids}")
        # Transform the anatomical structure IDs to include the full URI
        full_anatomical_ids = [f"http://purl.obolibrary.org/obo/{aid}" for aid in anatomical_ids]
        print(f"Transformed anatomical structure IDs: {full_anatomical_ids}")

        diseases = services.get_diseases_by_anatomical_structures(full_anatomical_ids)
        diseases = utils.convert_objectid_to_str(diseases)
        print(f"Found diseases: {diseases}")
        return create_json_response(jsonify(diseases), 200)
    
    @app.route("/diseases/by_exposures", methods=["POST"])
    def diseases_by_exposures():
        exposure_ids = request.json.get('exposure_ids')
        if not exposure_ids:
            return create_json_response(jsonify("exposure structure IDs are required"), 400)
        
        print(f"Received exposure structure IDs: {exposure_ids}")
        # Transform the exposure structure IDs to include the full URI
        full_exposure_ids = [f"http://purl.obolibrary.org/obo/{aid}" for aid in exposure_ids]
        print(f"Transformed exposure structure IDs: {full_exposure_ids}")

        diseases = services.get_diseases_by_exposures(full_exposure_ids)
        diseases = utils.convert_objectid_to_str(diseases)
        print(f"Found diseases: {diseases}")
        return create_json_response(jsonify(diseases), 200)
    
    @app.route("/diseases/by_treatment", methods=["POST"])
    def diseases_by_treatment():
        treatment_ids = request.json.get('treatment_ids')
        if not treatment_ids:
            return create_json_response(jsonify("treatment structure IDs are required"), 400)
        
        print(f"Received treatment structure IDs: {treatment_ids}")
        # Transform the treatment structure IDs to include the full URI
        full_treatment_ids = [f"http://purl.obolibrary.org/obo/{tid}" for tid in treatment_ids]
        print(f"Transformed treatment structure IDs: {full_treatment_ids}")

        diseases = services.get_diseases_by_treatments(full_treatment_ids)
        diseases = utils.convert_objectid_to_str(diseases)
        print(f"Found diseases: {diseases}")
        return create_json_response(jsonify(diseases), 200)
    
    @app.route("/diseases/by_chemicals", methods=["POST"])
    def diseases_by_chemicals():
        chemical_ids = request.json.get('chemical_ids')
        if not chemical_ids:
            return create_json_response(jsonify("chemical structure IDs are required"), 400)
        
        print(f"Received chemical structure IDs: {chemical_ids}")
        # Transform the chemical structure IDs to include the full URI
        full_chemical_ids = [f"http://purl.obolibrary.org/obo/{cid}" for cid in chemical_ids]
        print(f"Transformed chemical structure IDs: {full_chemical_ids}")

        diseases = services.get_diseases_by_chemicals(full_chemical_ids)
        diseases = utils.convert_objectid_to_str(diseases)
        print(f"Found diseases: {diseases}")
        return create_json_response(jsonify(diseases), 200)
    
    # TODO don't exclude the filters from the structures.
    @app.route("/diseases/by_filters", methods=["POST"])
    def diseases_by_filters():
        body = request.json
        phenotype_ids = body.get('phenotype_ids', [])
        anatomical_ids = body.get('anatomical_ids', [])
        age_onset_ids = body.get('age_onset_ids', [])
        exposure_ids = body.get('exposure_ids', [])
        treatment_ids = body.get('treatment_ids', [])
        chemical_ids = body.get('chemical_ids', [])
        # include_predictions = body.get('include_predictions', "true")
        
        # Transform the IDs to include the full URI
        full_phenotype_ids = [f"http://purl.obolibrary.org/obo/{pid}" for pid in phenotype_ids]
        full_anatomical_ids = [f"http://purl.obolibrary.org/obo/{aid}" for aid in anatomical_ids]
        full_age_onset_ids = [f"http://purl.obolibrary.org/obo/{aid}" for aid in age_onset_ids]
        full_exposure_ids = [f"http://purl.obolibrary.org/obo/{eid}" for eid in exposure_ids]
        full_treatment_ids = [f"http://purl.obolibrary.org/obo/{tid}" for tid in treatment_ids]
        full_chemical_ids = [f"http://purl.obolibrary.org/obo/{cid}" for cid in chemical_ids]

        diseases = services.get_diseases_by_filters(full_phenotype_ids, full_anatomical_ids,
                                                     full_age_onset_ids, full_exposure_ids, full_treatment_ids, full_chemical_ids)
        diseases = utils.convert_objectid_to_str(diseases)
        return create_json_response(jsonify(diseases), 200)

    @app.route("/phenotypes", methods=["GET"])
    def get_phenotypes():
        return create_json_response(jsonify(services.get_phenotypes()), 200)
    
    @app.route("/anatomical_structures", methods=["GET"])
    def get_anatomical_structures():
        return create_json_response(jsonify(services.get_anatomical_structures()), 200)

    @app.route("/age_onsets", methods=["GET"])
    def get_age_onsets():
        return create_json_response(jsonify(services.get_age_onsets()), 200)
    
    @app.route("/exposures", methods=["GET"])
    def get_exposures():
        return create_json_response(jsonify(services.get_exposures()), 200)
    
    @app.route("/treatments", methods=["GET"])
    def get_etreatments():
        return create_json_response(jsonify(services.get_treatments()), 200)
    
    @app.route("/chemicals", methods=["GET"])
    def get_chemicals():
        return create_json_response(jsonify(services.get_chemicals()), 200)
    
    @app.route("/relationship_types", methods=["GET"])
    def get_relationship_types():
        return create_json_response(jsonify(services.get_relationship_types()), 200)
    

    @app.route("/diseases/predict_relationship", methods=["POST"])
    def predict_relationship():

        # new_disease_id = "MONDO_0006781"
        # new_relationship_type = "has_relationship"
        # new_relationship_property = "RO_0004027"
        body = request.json
        disease_id = body.get('disease_id', '')
        new_relationship_type = body.get('new_relationship_type', 'has_relationship')
        new_relationship_property = body.get('new_relationship_property', [])

        # Transform the IDs to include the full URI
        full_disease_id = f"http://purl.obolibrary.org/obo/{disease_id}"
        full_new_relationship_property = f"http://purl.obolibrary.org/obo/{new_relationship_property}"

        # TODO default phenotype
        relationship_type = services.get_relationship_type_for_property(full_new_relationship_property)
        
        if relationship_type == constants.UBERON_STR:
            model_name = 'anatomical'
        elif relationship_type == constants.HP_STR:
            model_name = 'phenotypes'
        else:
            return jsonify({'error': 'Invalid relationship type'}), 400

        predicted_target = services.predict_relationship(full_disease_id, new_relationship_type, full_new_relationship_property, model_name)
        print(f'Predicted target: {predicted_target}')

        services.update_data_model(full_disease_id, full_new_relationship_property, predicted_target)
        
        return create_json_response(jsonify(predicted_target), 200) 

    ################## DEBUG
    @app.route('/diseases/seen_labels', methods=['GET'])
    def get_seen_labels():
        with repository.fs.get_last_version('seen_labels.json') as file_data:
            seen_labels = json.loads(file_data.read().decode('utf-8'))
        return create_json_response(jsonify(seen_labels), 200)
    

def create_json_response(data, status_code=200):
    response = make_response(data, status_code)
    response.headers['Content-Type'] = 'application/json'
    return response

def load_model_from_grid_fs(model_name):
    with tempfile.NamedTemporaryFile() as temp_file:
        model_file = repository.fs.find_one({'filename': f'{model_name}.pkl'})
        temp_file.write(model_file.read())
        temp_file.flush()
        return poplib.load(temp_file.name)