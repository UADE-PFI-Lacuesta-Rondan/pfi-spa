import json
import tempfile
import services
import constants

import repository
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from imblearn.over_sampling import RandomOverSampler
#from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from imblearn.over_sampling import SMOTE

def generate_model():

    diseases = repository.get_diseases()
    
    # cluster characteristic
    df_with_clusters = clustering_data_frame()

    anatomical_df = get_data_frame(diseases, 'anatomical_structures', constants.UBERON_STR)
    phenotypes_df = get_data_frame(diseases, 'phenotypes', constants.HP_STR)
    #exposures_df = get_data_frame(diseases, 'exposures', constants.ECTO_STR)
    #treatments_df = get_data_frame(diseases, 'treatments', constants.MAXO_STR)

    # generate models with and without clusters
    generate_model_for_type(anatomical_df, 'anatomical', df_with_clusters, include_cluster=False)
    generate_model_for_type(phenotypes_df, 'phenotypes', df_with_clusters, include_cluster=False)
    #generate_model_for_type(exposures_df, 'exposures', df_with_clusters, include_cluster=False)
    #generate_model_for_type(treatments_df, 'treatments', df_with_clusters, include_cluster=False)

    generate_model_for_type(anatomical_df, 'anatomical', df_with_clusters, include_cluster=True)
    generate_model_for_type(phenotypes_df, 'phenotypes', df_with_clusters, include_cluster=True)
    #generate_model_for_type(exposures_df, 'exposures', df_with_clusters, include_cluster=True)
    #generate_model_for_type(treatments_df, 'treatments', df_with_clusters, include_cluster=True)

def get_data_frame(diseases, relationship_type, type_key):
    records = []
    for disease in diseases:
        disease_id = disease['id']
        for relationship in disease.get(relationship_type, []):
            property_id = relationship['property']
            target_id = relationship['target']
            if not services.is_valid_relationship(property_id, target_id):
                continue
            records.append({
                'disease_id': disease_id,
                'relationship_property': relationship['property'],
                'target_id': relationship['target']
            })

    df = pd.DataFrame(records)
    


    return df

def generate_model_for_type(df, model_name, df_with_clusters, include_cluster=False):
    """
    Generate specialized RandomForest model using previously generated collections of diseases and data models,
    and save the trained model and associated encoders to MongoDB.

    Hyperparameters:
    - n_estimators: Number of trees in the forest. Randomly chosen between 10 and 30. High number of n_estimators complexizes the model, giving better accuracy but reduced performance.
    - max_depth: Maximum depth of the trees. Chosen from [10, 20, None]. Maximum depth of the tree will be 10, 20 or None. Nodes are expanded until all leaves are pure or until they contain fewer than min_samples_split samples.
    - min_samples_split: Minimum number of samples required to split an internal node. Randomly chosen between 2 and 4.
    - min_samples_leaf: Minimum number of samples required to be at a leaf node. Randomly chosen between 1 and 2.
    """

    df['disease_id'] = df['disease_id'].astype(str)
    df['relationship_property'] = df['relationship_property'].astype(str)
    df['target_id'] = df['target_id'].astype(str)
    
    df_with_clusters['Disease ID'] = df_with_clusters['Disease ID'].astype(str)
    df_with_clusters['Property'] = df_with_clusters['Property'].astype(str)
    df_with_clusters['Target'] = df_with_clusters['Target'].astype(str)

    # merge dataframe with clustered dataframe
    df = pd.merge(df, df_with_clusters[['Disease ID', 'Property', 'Target', 'Cluster']],
                  left_on=['disease_id', 'relationship_property', 'target_id'],
                  right_on=['Disease ID', 'Property', 'Target'],
                  how='left')
    
    # Drop unnecessary columns after merge
    df.drop(columns=['Disease ID', 'Property', 'Target'], inplace=True)

    # categorical features (disease_id, relationship_property, target_id) are encoded using LabelEncoder.
    le_disease = LabelEncoder()
    le_relationship_property = LabelEncoder()
    le_target_id = LabelEncoder()

    df['disease_id'] = le_disease.fit_transform(df['disease_id'])
    df['relationship_property'] = le_relationship_property.fit_transform(df['relationship_property'])
    df['target_id'] = le_target_id.fit_transform(df['target_id'])

    #print(df['disease_id'].value_counts())
    #print(df['relationship_property'].value_counts())
    #print(df['target_id'].value_counts())

    # disease_rel_prop characteristic is created by combining disease_id and relationship_property.
    df['disease_rel_prop'] = df['disease_id'].astype(str) + '_' + df['relationship_property'].astype(str)
    le_disease_rel_prop = LabelEncoder()
    df['disease_rel_prop'] = le_disease_rel_prop.fit_transform(df['disease_rel_prop'])

    X = df[['disease_id', 'relationship_property', 'disease_rel_prop']]
    if include_cluster:
        le_cluster = LabelEncoder()
        df['Cluster'] = le_cluster.fit_transform(df['Cluster'].fillna(-1))  # fillna for potential NaNs
        X['Cluster'] = df['Cluster']

    y = df['target_id']

    if len(X) < 10:
        print(f"Insufficient data to train {model_name} model.")
        return
    
    #reduce data_set
    #X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # data augmentation
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    min_samples_per_class = min(pd.Series(y_train_res).value_counts())
    n_splits = max(2, min(3, min_samples_per_class))

    if min_samples_per_class < 2:
        print(f"Insufficient data to train {model_name} model with at least 2 splits.")
        return

    rf = RandomForestClassifier()
    param_dist = {
        'n_estimators': randint(10, 30),
        'max_depth': [10, 20, None],
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 3),
    }

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=n_splits, n_jobs=1, verbose=0, random_state=42)
    random_search.fit(X_train_res, y_train_res)

    best_rf = random_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {model_name}: {accuracy}')
    #print('Classification Report:')
    #print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=le_target_id.inverse_transform(np.unique(y_test))))

    model_files = {
        f'{model_name}.pkl': best_rf,
        f'le_disease_{model_name}.pkl': le_disease,
        f'le_relationship_property_{model_name}.pkl': le_relationship_property,
        f'le_target_id_{model_name}.pkl': le_target_id,
        f'le_disease_rel_prop_{model_name}.pkl': le_disease_rel_prop,
    }

    if include_cluster:
        model_files[f'le_cluster_{model_name}.pkl'] = le_cluster

    for filename, model in model_files.items():
        with tempfile.NamedTemporaryFile() as temp_file:
            joblib.dump(model, temp_file.name)
            with open(temp_file.name, 'rb') as file_data:
                repository.fs.put(file_data, filename=filename)
    
    seen_labels = {
       f'le_disease_{model_name}': le_disease.classes_.tolist(),
       f'le_relationship_property_{model_name}': le_relationship_property.classes_.tolist(),
       f'le_target_id_{model_name}': le_target_id.classes_.tolist(),
       f'le_disease_rel_prop_{model_name}': le_disease_rel_prop.classes_.tolist(),
    }

    if include_cluster:
        seen_labels[f'le_cluster_{model_name}'] = le_cluster.classes_.tolist()

    repository.fs.put(json.dumps(seen_labels).encode('utf-8'), filename=f'seen_labels_{model_name}.json')

# Function to add relationships to the data
def add_relationships(disease_data, disease, rel_type):
    disease_id = disease.get('id')
    name = disease.get('name')
    description = disease.get('description')
    for relationship in disease[rel_type]: # phenotypes, anatomical_structures, chemicals, etc
        property_ = relationship.get('property')
        target = relationship.get('target')
        disease_data.append([disease_id, name, description, rel_type, property_, target])

def get_dbscan_clustering_data_frame(): 
    data = repository.get_diseases()
    disease_data = []

    # relationships: phenotypes, chemicals, and anatomical_structures, etc
    for disease in data:  
        if 'phenotypes' in disease:
            add_relationships(disease_data, disease, 'phenotypes')
        if 'chemicals' in disease:
            add_relationships(disease_data, disease, 'chemicals')
        if 'anatomical_structures' in disease:
            add_relationships(disease_data, disease, 'anatomical_structures')

    columns = ['Disease ID', 'Name', 'Description', 'Relationship Type', 'Property', 'Target']
    df = pd.DataFrame(disease_data, columns=columns)

    return df

def clustering_data_frame(): 

    df = get_dbscan_clustering_data_frame()
    # drop if exists, for now
    if 'Description' in df.columns:
        df = df.drop(columns=['Description'])
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    # before removing duplicates
    count_before = len(df)

    # duplicates
    df = df.drop_duplicates()

    #after removing duplicates
    count_after = len(df)
    print(f"Count before removing duplicates: {count_before}")
    print(f"Count after removing duplicates: {count_after}")

    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)

    # drop rows with any missing values
    df = df.dropna()

    # One-hot encode columns
    df = pd.get_dummies(df, columns=['Relationship Type'], prefix='RelType')

    df.head()

    # select features for clustering
    features = ['Disease ID', 'Property', 'Target', 'RelType_anatomical_structures', 'RelType_chemicals', 'RelType_phenotypes']

    # encode categorical features
    df['Disease ID'] = df['Disease ID'].astype('category').cat.codes
    df['Property'] = df['Property'].astype('category').cat.codes
    df['Target'] = df['Target'].astype('category').cat.codes

    # scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # apply DBSCAN
    eps_value = 0.9 
    min_samples_value = 2 * len(features)

    dbscan_clusterer = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    clusters = dbscan_clusterer.fit_predict(scaled_features)

    # Add cluster labels to the dataframe
    df['Cluster'] = clusters

    # Calculate the number of clusters and noise points
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f'Number of clusters: {n_clusters}')
    print(f'Number of noise points: {n_noise}')

    # PCA for visualization
    #pca = PCA(n_components=2)
    #pca_result = pca.fit_transform(scaled_features)

    # Plot the clusters
    #plt.figure(figsize=(10, 7))
    #plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k')
    #plt.title('DBSCAN Clustering with PCA')
    #plt.xlabel('PCA Component 1')
    #plt.ylabel('PCA Component 2')
    #plt.colorbar(label='Cluster Label')
    #plt.show()

    # silhouette score calculation based on scaled_features and computed clusters
    sil_score = silhouette_score(scaled_features, clusters)
    print(f'Silhouette Score: {sil_score:.2f}')

    return df

