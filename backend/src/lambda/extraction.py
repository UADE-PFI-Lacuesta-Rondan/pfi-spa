import boto3
import requests

def lambda_handler(event, context):

    # get last release from hithub repo
    # ref https://docs.github.com/en/rest/releases/releases?apiVersion=2022-11-28#get-the-latest-release
    repo_owner = "monarch-initiative"
    repo_name = "mondo"
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
    
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        release_data = response.json()
        tag_name = release_data['tag_name']
        
        # build uri to ontology in format json
        file_url = f"https://github.com/{repo_owner}/{repo_name}/releases/download/{tag_name}/mondo.json"
        
        # download ontology
        file_response = requests.get(file_url)
        if file_response.status_code == 200:
            
            # load file to ontology-bucket
            s3 = boto3.client('s3')
            bucket_name = 'ontology-bucket' # TODO bucket name
            file_name = 'mondo.json'
            
            s3.put_object(Bucket=bucket_name, Key=file_name, Body=file_response.content)
            return {
                'statusCode': 200,
                'body': f'Archivo {file_name} cargado correctamente a S3 en el bucket {bucket_name}.'
            }
        else:
            return {
                'statusCode': file_response.status_code,
                'body': 'Error al descargar el archivo mondo.json.'
            }
    else:
        return {
            'statusCode': response.status_code,
            'body': 'Error al obtener la ultima release de github.'
        }
