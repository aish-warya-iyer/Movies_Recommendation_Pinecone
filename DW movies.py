from airflow import DAG
from datetime import datetime, timedelta
from airflow.decorators import task
from airflow.models import Variable
import pandas as pd
import time
import requests
import os

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='Movies_to_Pinecone',
    default_args=default_args,
    description='Movie Search Engine using Pinecone',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=['movies', 'pinecone', 'search-engine'],
) as dag:
    
    @task
    def download_data():
        """Download updated movie dataset"""
        data_dir = '/tmp/movie_data'
        os.makedirs(data_dir, exist_ok=True)
        file_path = f"{data_dir}/tmdb_5000_movies.csv"
        
        url = 'https://grepp-reco-test.s3.ap-northeast-2.amazonaws.com/tmdb_5000_movies.csv'
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded movie data to: {file_path}")
        else:
            raise Exception(f"❌ Failed to download data: HTTP {response.status_code}")
        
        return file_path

    @task
    def preprocess_data(data_path):
        """Clean and format metadata for embedding"""
        df = pd.read_csv(data_path)
        df['title'] = df['title'].astype(str).fillna('')
        df['overview'] = df['overview'].astype(str).fillna('')
        df['metadata'] = df.apply(lambda row: {'title': f"{row['title']} {row['overview']}"}, axis=1)
        df['id'] = df.reset_index(drop=True).index.astype(str)
        
        preprocessed_path = '/tmp/movie_data/movies_preprocessed.csv'
        df.to_csv(preprocessed_path, index=False)
        print(f"✅ Preprocessed data saved to: {preprocessed_path}")
        return preprocessed_path

    @task
    def create_pinecone_index():
        """Initialize Pinecone index"""
        api_key = Variable.get("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)

        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        index_name = 'movie-search-index'

        if index_name in [idx['name'] for idx in pc.list_indexes()]:
            pc.delete_index(index_name)
            print(f"ℹ️ Existing index '{index_name}' deleted.")

        pc.create_index(
            name=index_name,
            dimension=384,
            metric='dotproduct',
            spec=spec
        )

        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        print(f"✅ Pinecone index '{index_name}' is ready.")
        return index_name

    @task
    def generate_embeddings_and_upsert(data_path, index_name):
        """Generate sentence embeddings and upload to Pinecone"""
        api_key = Variable.get("PINECONE_API_KEY")
        df = pd.read_csv(data_path)
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()
            metadata_list = batch_df['metadata'].apply(eval).tolist()
            texts = [meta['title'] for meta in metadata_list]
            embeddings = model.encode(texts)
            
            upsert_data = [
                {
                    'id': str(row['id']),
                    'values': embeddings[j].tolist(),
                    'metadata': metadata_list[j]
                }
                for j, (_, row) in enumerate(batch_df.iterrows())
            ]
            index.upsert(upsert_data)
            print(f"✅ Upserted batch {i // batch_size + 1} to index.")
        
        return index_name

    @task
    def test_search_query(index_name):
        """Run a semantic query and print top matches"""
        api_key = Variable.get("PINECONE_API_KEY")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        query = "space adventure"
        query_embedding = model.encode(query).tolist()

        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        print(f"\n🔍 Search results for query: '{query}':")
        for match in results['matches']:
            print(f"> {match['metadata']['title'][:80]}... (score: {match['score']:.2f})")

    # Define tasks
    download_task = download_data()
    preprocess_task = preprocess_data(download_task)
    create_index_task = create_pinecone_index()
    upsert_task = generate_embeddings_and_upsert(preprocess_task, create_index_task)
    search_task = test_search_query(upsert_task)

    # Set dependencies to match your desired flow
    download_task >> preprocess_task
    preprocess_task >> upsert_task
    create_index_task >> upsert_task
    upsert_task >> search_task