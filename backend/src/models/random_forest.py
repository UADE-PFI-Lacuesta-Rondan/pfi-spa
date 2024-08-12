import json
import tempfile
import services
import constants

import repository
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import randint
import joblib

def get_data_frame(): 

    # get data structures previously generated collections
    diseases = repository.get_diseases()
    data_model = repository.get_data_model()

    records = []

    for disease in diseases:
        disease_id = disease['id']
        disease_name = disease['name']
    
        # treatment relationships  
        for treatment in disease.get('treatments', []):

            property_id = treatment['property']
            target_id = treatment['target']

            if not services.is_valid_relationship(property_id, target_id):
                continue

            records.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'relationship_type': treatment['type'],
                'relationship_property': treatment['property'],
                'target_id': treatment['target']
            })

        # anatomical relationships
        for anatomical in disease.get('anatomical_structures', []):

            property_id = anatomical['property']
            target_id = anatomical['target']

            if not services.is_valid_relationship(property_id, target_id):
                continue

            records.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'relationship_type': anatomical['type'],
                'relationship_property': anatomical['property'],
                'target_id': anatomical['target']
            })

        # phenotypic relationships
        for phenotype in disease.get('phenotypes', []):

            property_id = phenotype['property']
            target_id = phenotype['target']

            if not services.is_valid_relationship(property_id, target_id):
                continue

            records.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'relationship_type': phenotype['type'],
                'relationship_property': phenotype['property'],
                'target_id': phenotype['target']
            })

        # age onset relationships
        for age_onset in disease.get('age_onsets', []):

            property_id = age_onset['property']
            target_id = age_onset['target']

            if not services.is_valid_relationship(property_id, target_id):
                continue

            records.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'relationship_type': age_onset['type'],
                'relationship_property': age_onset['property'],
                'target_id': age_onset['target']
            })

        # exposure relationships
        for exposure in disease.get('exposures', []):

            property_id = exposure['property']
            target_id = exposure['target']

            if not services.is_valid_relationship(property_id, target_id):
                continue

            records.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'relationship_type': exposure['type'],
                'relationship_property': exposure['property'],
                'target_id': exposure['target']
            })

        # chemical relationships
        for chemical in disease.get('chemicals', []):

            property_id = chemical['property']
            target_id = chemical['target']

            if not services.is_valid_relationship(property_id, target_id):
                continue

            records.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'relationship_type': chemical['type'],
                'relationship_property': chemical['property'],
                'target_id': chemical['target']
            })
    df = pd.DataFrame(records)
    return df

def generate_model():
    """
    Generate a RandomForest model using previously generated collections of diseases and data models,
    and save the trained model and associated encoders to MongoDB.

    This function performs the following steps:
    1. Fetches data from MongoDB collections for diseases and data models.
    2. Prepares the data for training by converting it into a suitable format.
    3. Encodes categorical features and creates interaction features for better modeling.
    4. Trains a RandomForest model using RandomizedSearchCV for hyperparameter optimization.
    5. Evaluates the trained model on a test set.
    6. Saves the trained model and encoders to MongoDB for future use.

    Hyperparameters:
    - n_estimators: Number of trees in the forest. Randomly chosen between 10 and 30. High number of n_estimators complexizes the model, giving better accuracy but reduced performance.
    - max_depth: Maximum depth of the trees. Chosen from [10, 20, None]. Maximum depth of the tree will be 10, 20 or None. Nodes are expanded until all leaves are pure or until they contain fewer than min_samples_split samples.
    - min_samples_split: Minimum number of samples required to split an internal node. Randomly chosen between 2 and 4.
    - min_samples_leaf: Minimum number of samples required to be at a leaf node. Randomly chosen between 1 and 2.

    Feature Engineering:
    - Categorical features (disease_id, relationship_type, relationship_property, target_id) are encoded using LabelEncoder.
    - An interaction feature (disease_rel_prop) is created by combining disease_id and relationship_property.
    """
    
    df = get_data_frame()

    # categorical data encoding
    le_disease = LabelEncoder()
    le_relationship_type = LabelEncoder()
    le_relationship_property = LabelEncoder()
    le_target_id = LabelEncoder()

    df['disease_id'] = le_disease.fit_transform(df['disease_id'])
    df['relationship_type'] = le_relationship_type.fit_transform(df['relationship_type'])
    df['relationship_property'] = le_relationship_property.fit_transform(df['relationship_property'])
    df['target_id'] = le_target_id.fit_transform(df['target_id'])

    # disease_rel_prop interaction characteristic
    df['disease_rel_prop'] = df['disease_id'].astype(str) + '_' + df['relationship_property'].astype(str)
    le_disease_rel_prop = LabelEncoder()
    df['disease_rel_prop'] = le_disease_rel_prop.fit_transform(df['disease_rel_prop'])

    X = df[['disease_id', 'relationship_type', 'relationship_property', 'disease_rel_prop']]
    y = df['target_id']

    # reduce dataset
    #X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)

    # split dataset into test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomOverSampler balance clases with randomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # define RandomForest model
    rf = RandomForestClassifier()

    # hyper parameters definition
    param_dist = {
        'n_estimators': randint(10, 30),
        'max_depth': [10, 20, None],
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 3),
    }

    # RandomizedSearchCV parameters, less iterators single worker
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=1, verbose=2, random_state=42)
    random_search.fit(X_train_res, y_train_res)

    # get RandomForest best indicator
    best_rf = random_search.best_estimator_

    # predict in X test subset
    y_pred = best_rf.predict(X_test)

    # unique labels in test subset
    unique_labels = np.unique(y_test)

    # evaluate model TODO needs improvement
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=le_target_id.inverse_transform(unique_labels)))

    # save random_forest model to MongoDB
    model_files = {
        'best_rf.pkl': best_rf,
        'le_disease.pkl': le_disease,
        'le_relationship_type.pkl': le_relationship_type,
        'le_relationship_property.pkl': le_relationship_property,
        'le_target_id.pkl': le_target_id,
        'le_disease_rel_prop.pkl': le_disease_rel_prop
    }

    # seen labels
    seen_labels = {
       'le_disease': le_disease.classes_.tolist(),
       'le_relationship_type': le_relationship_type.classes_.tolist(),
       'le_relationship_property': le_relationship_property.classes_.tolist(),
       'le_target_id': le_target_id.classes_.tolist(),
       'le_disease_rel_prop': le_disease_rel_prop.classes_.tolist()
    }

    for filename, model in model_files.items():
        with tempfile.NamedTemporaryFile() as temp_file:
            joblib.dump(model, temp_file.name)
            with open(temp_file.name, 'rb') as file_data:
                repository.fs.put(file_data, filename=filename)
    
    # save seen labels to mongoDB FS
    repository.fs.put(json.dumps(seen_labels).encode('utf-8'), filename='seen_labels.json')