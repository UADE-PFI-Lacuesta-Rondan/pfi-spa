import pandas as pd
import repository
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

def get_data_frame(): 
    data = repository.get_diseases()
    disease_data = []

    for disease in data:
        disease_id = disease.get('id')
        name = disease.get('name')
        description = disease.get('description')

        # Function to add relationships to the data
        def add_relationships(rel_type, relationships):
            for relationship in relationships:
                property_ = relationship.get('property')
                target = relationship.get('target')
                disease_data.append([disease_id, name, description, rel_type, property_, target])

        # Add relationships, phenotypes, chemicals, and anatomical_structures
        if 'phenotypes' in disease:
            add_relationships('phenotypes', disease['phenotypes'])
        if 'chemicals' in disease:
            add_relationships('chemicals', disease['chemicals'])
        if 'anatomical_structures' in disease:
            add_relationships('anatomical_structures', disease['anatomical_structures'])

    columns = ['Disease ID', 'Name', 'Description', 'Relationship Type', 'Property', 'Target']
    df = pd.DataFrame(disease_data, columns=columns)
    return df

def preprocess_data_model(): 

    df = get_data_frame()
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
    output_file = 'preprocessed_disease_data.csv'
    df.to_csv(output_file, index=False)

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
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k')
    plt.title('DBSCAN Clustering with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()

    # Assuming `scaled_features` is your data and `clusters` are the labels obtained from clustering
    sil_score = silhouette_score(scaled_features, clusters)
    print(f'Silhouette Score: {sil_score:.2f}')


