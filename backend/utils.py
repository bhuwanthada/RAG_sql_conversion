import json
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
import os
import numpy as np
import psycopg2
import sqlite3
from google.cloud import aiplatform
from langchain.llms import VertexAI
# from sentence_transformers import SentenceTransformer
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from backend.vector_db import VectorDB
from langchain.vectorstores import FAISS
import uuid
from google import genai
from google.genai import types
from backend.vector_db import VectorDB
import re
import pandas as pd
from random import randint

load_dotenv()

app = FastAPI()

# Database Configuration (Move to environment variables for production)
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
PROJECT_ID=os.getenv("project", "")
LOCATION=os.getenv("location", "")
# model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
model_embedding = SentenceTransformer('all-mpnet-base-v2')

# Vertex AI Configuration
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Path to your service account key
MODEL_NAME = "gemini-2.0-flash-001"  # Or your preferred text model

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)


# Database Connection Function
def get_db_connection():
    try:
        conn = sqlite3.connect("ecommerce.db")
        cursor = conn.cursor()
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to the database")


# Data Models


# Initialize Langchain SQLDatabase and SQLDatabaseChain
# def create_db_chain(conn):
#     db = SQLDatabase.from_uri(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
#     llm = VertexAI(model_name=MODEL_NAME, temperature=0.0)  # Adjust temperature as needed
#     db_chain = SQLDatabaseChain.from_llm(llm, conn, verbose=True)  # Set verbose=True for debugging
#     return db_chain


def get_database_schema(conn):
    """Extract table names, column names, and column types from the database."""
    schema_dict = {}
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # table_names = [row[0] for row in cursor.fetchall()]
    # cursor.execute("""
    #     SELECT
    #         table_name,
    #         column_name,
    #         data_type
    #     FROM
    #         information_schema.columns
    #     WHERE
    #         table_schema = 'public'  -- Assuming you're using the 'public' schema
    #     ORDER BY
    #         table_name,
    #         ordinal_position;
    # """)
    schema_info = cursor.fetchall()
    for _entity in schema_info:
        if _entity[0] != "sqlite_sequence":
            schema_dict[_entity[0]] = []
            cursor.execute(f"PRAGMA table_info('{_entity[0]}')")
            schema = cursor.fetchall()
            for table_entity in schema:
                schema_dict[_entity[0]].append({"column_name": table_entity[1], "column_data_type": table_entity[2]})
    # print(schema_info)
    # cursor.execute(f"PRAGMA table_info('Customer')")
    # cursor.close()
    #
    # # Format the schema information into a usable string or dictionary
    # schema_dict = {}
    # for table_name, column_name, data_type in schema_info:
    #     if table_name not in schema_dict:
    #         schema_dict[table_name] = []
    #     schema_dict[table_name].append({"column_name": column_name, "data_type": data_type})
    return schema_dict


def schema_to_text(schema_dict):
    """Convert the schema dictionary into a readable text format."""
    schema_text = ""
    for table_name, columns in schema_dict.items():
        schema_text += f"Table: {table_name}\n"
        for column in columns:
            schema_text += f"  - {column['column_name']} ({column['column_data_type']})\n"
    return schema_text


def schema_to_text_with_desc(schema_dict):
    """Convert the schema dictionary into a readable text format."""
    schema_text = ""
    for table_name, columns in schema_dict.items():
        schema_text += f"Table: {table_name}\n"
        for column in columns:
            schema_text += f"  - {column['column_name']} ({column['column_data_type']}) and description: {column['description']}\n"
    return schema_text


def create_chroma_vector_database(schema_dict, embed_name):  # Or "text-embedding-ada-002"
    """Creates a FAISS vector database from the database schema."""
    # Create text embeddings for each table and column description
    documents = []
    metadatas = []
    for table_name, columns in schema_dict.items():
        table_description = f"Table: {table_name}"
        documents.append(table_description)
        metadatas.append({"type": "table", "name": table_name})

        for column in columns:
            column_description = f"Column: {column['column_name']} in table {table_name} (Type: {column['column_data_type']})"
            documents.append(column_description)
            metadatas.append({"type": "column", "table": table_name, "name": column['column_name']})
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    vector_db = VectorDB()
    collection_obj = vector_db.get_or_create_collection(collection_name=embed_name)
    vector_db.add_data(collection_obj, documents, metadatas, ids)
    print("Embedding created successfully using chromadb..!!")


def create_json_to_chroma_vector_db(metadata, embed_name):
    documents = []
    metadatas = []
    for table in metadata:
        cols_desc = []
        for column in table['columns']:
            cols_desc.append({"column_name": column['columns_name'], "data type": column['column_data_type'],
                              "description": column['description']})
        documents.append(json.dumps({"table": table['table_name'], "description": table['description']}))
        metadatas.append({"type_one": "Table", "description_one": table['table_name'],
                          "type_two": "column", "description_two": cols_desc})
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    vector_db = VectorDB()
    collection_obj = vector_db.get_or_create_collection(collection_name=embed_name)
    vector_db.add_data(collection_obj, documents, metadatas, ids)
    print("Embedding created successfully using chromadb..!!")


def generate_sql_using_json_chromadb(query, embed_name):  # Or any other suitable LLM
    """Generates SQL query using a Large Language Model (LLM)."""

    # 1. Similarity Search
    vector_db = VectorDB()
    collection_obj = vector_db.get_or_create_collection(collection_name=embed_name)
    relevant_schema = vector_db.query(collection_obj, [query])
    context = ""
    for i in range(len(relevant_schema['documents'][0])):
        context += f"{relevant_schema['documents'][0][i]} and supportive other details such as: {relevant_schema['metadatas'][0][i]}" + "\n"

    table_context = ""
    for table in relevant_schema['documents'][0]:
        table_context += f"Table Name: {table['table_name']}\n"
        table_context += f"Description: {table['description']}\n"
        table_context += "Columns:\n"
        for column in table['columns']:
            table_context += f"  - {column['columns_name']}: {column['description']} (Data Type: {column['column_data_type']})\n"
        table_context += "\n"

    # 2. Augment Prompt
    # context = "\n".join([doc.page_content for doc in relevant_schema])
    # bkup_prompt = f"""
    # You are a SQL expert. You are given a database schema and a user query.
    # Generate the SQL query that answers the user query.
    #
    # Database Schema:
    # {db_schema}
    #
    # Relevant parts of the schema (based on similarity to the query):
    # {context}
    #
    # User Query: {query}
    #
    # SQL Query:
    # """
    prompt = f"""
        You are a SQL expert having decades of experience. You are given a user query and context (based on similarity to the query).
        You have to strictly stick with the provided information and generate the result.
        You will be rewarded with 100 bucks in case of providing correct answer while you will be 
        heavily penalized in case of providing wrong answer. You have to provide sql query only. Do not add any other 
        information in the output

        Generate the SQL query that answers the user query.

        context (based on similarity to the query):
        {context}

        User Query: {query}

        SQL Query:
        """
    # GEMINI CALL.
    client = genai.Client(
        vertexai=True,
        project=os.getenv("project", ""),
        location=os.getenv("location", ""),
    )
    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
    )
    resp = client.models.generate_content(model=model, contents=contents, config=generate_content_config)
    if resp.text:
        generated_sql_query = resp.text
        match = re.search(r"```sql\s*(.*?)\s*```", generated_sql_query,
                          re.DOTALL)  # Use re.DOTALL to match across newlines
        if match:
            print(f"match: {match}")
            return match.group(1).strip().replace("\n",
                                                  " ")  # Return the captured group (the SQL query) and strip whitespace
        else:
            return generated_sql_query

    else:
        "No sql query generated."


def create_chroma_vector_database_with_metadata(schema_dict):  # Or "text-embedding-ada-002"
    """Creates a FAISS vector database from the database schema."""
    # Create text embeddings for each table and column description
    documents = []
    metadatas = []
    for table_name, columns in schema_dict.items():
        table_description = f"Table: {table_name}"
        documents.append(table_description)
        metadatas.append({"type": "table", "name": table_name})

        for column in columns:
            column_description = f"Column: {column['column_name']} in table {table_name} (Type: {column['column_data_type']})"
            documents.append(column_description)
            metadatas.append({"type": "column", "table": table_name, "name": column['column_name'],
                              "description": column['description']})
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    vector_db = VectorDB()
    collection_obj = vector_db.get_or_create_collection(collection_name="ecommerce_embeddings")
    vector_db.add_data(collection_obj, documents, metadatas, ids)
    print("Embedding created successfully using chromadb..!!")


def create_faiss_vector_database(schema_dict, embedding_model_name="all-mpnet-base-v2"):  # Or "text-embedding-ada-002"
    """Creates a FAISS vector database from the database schema."""
    # Create text embeddings for each table and column description
    documents = []
    metadatas = []
    for table_name, columns in schema_dict.items():
        table_description = f"Table: {table_name}"
        documents.append(table_description)
        metadatas.append({"type": "table", "name": table_name})

        for column in columns:
            column_description = f"Column: {column['column_name']} in table {table_name} (Type: {column['column_data_type']})"
            documents.append(column_description)
            metadatas.append({"type": "column", "table": table_name, "name": column['column_name']})
    # embeddings = VertexAIEmbeddings(model_name=embedding_model_name, project="pe-rnd", location="us-central1")
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest", project="pe-rnd", location="us-central1")

    # Create the FAISS vector database
    db = FAISS.from_texts(documents, embeddings, metadatas=metadatas)
    return db


def generate_sql_using_faiss(query, db_schema, db):  # Or any other suitable LLM
    """Generates SQL query using a Large Language Model (LLM)."""

    # 1. Similarity Search
    relevant_schema = db.similarity_search(query, k=4)  # Adjust 'k' as needed

    # 2. Augment Prompt
    context = "\n".join([doc.page_content for doc in relevant_schema])
    prompt = f"""
    You are a SQL expert. You are given a database schema, context(based upon similarity to the query) and a user query.
    Generate the SQL query that answers the user query.
    You have to strictly stick with the provided information and generate the result.
    You will be rewarded with 100 bucks in case of providing correct answer while you will be 
    heavily penalized in case of providing wrong answer. You have to provide sql query only. Do not add any other 
    information in the output

    Generate the SQL query that answers the user query.

    Database Schema:
    {db_schema}

    Relevant parts of the schema (based on similarity to the query):
    {context}

    User Query: {query}

    SQL Query:
    """
    # GEMINI CALL.
    client = genai.Client(
        vertexai=True,
        project=os.getenv("project", ""),
        location=os.getenv("location", ""),
    )
    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
    )
    resp = client.models.generate_content(model=model, contents=contents, config=generate_content_config)
    if resp.text:
        generated_sql_query = resp.text
        match = re.search(r"```sql\s*(.*?)\s*```", generated_sql_query,
                          re.DOTALL)  # Use re.DOTALL to match across newlines
        if match:
            return match.group(1).strip()  # Return the captured group (the SQL query) and strip whitespace
        else:
            return generated_sql_query
    else:
        "No sql query generated."


def generate_sql_using_chromadb(query, embed_name):  # Or any other suitable LLM
    """Generates SQL query using a Large Language Model (LLM)."""

    # 1. Similarity Search
    vector_db = VectorDB()
    collection_obj = vector_db.get_or_create_collection(collection_name=embed_name)
    relevant_schema = vector_db.query(collection_obj, [query])
    print(f"relevant_schema: {relevant_schema}")
    context = ""
    for i in range(len(relevant_schema['documents'][0])):
        context += f"{relevant_schema['documents'][0][i]} and supportive other details such as: {relevant_schema['metadatas'][0][i]}" + "\n"
    print(f"context: {context}")

    # 2. Augment Prompt
    # context = "\n".join([doc.page_content for doc in relevant_schema])
    # bkup_prompt = f"""
    # You are a SQL expert. You are given a database schema and a user query.
    # Generate the SQL query that answers the user query.
    #
    # Database Schema:
    # {db_schema}
    #
    # Relevant parts of the schema (based on similarity to the query):
    # {context}
    #
    # User Query: {query}
    #
    # SQL Query:
    # """
    prompt = f"""
        You are a SQL expert having decades of experience. You are given a user query and context (based on similarity to the query).
        You have to strictly stick with the provided information and generate the result.
        You will be rewarded with 100 bucks in case of providing correct answer while you will be 
        heavily penalized in case of providing wrong answer. You have to provide sql query only. Do not add any other 
        information in the output

        Generate the SQL query that answers the user query.

        context (based on similarity to the query):
        {context}

        User Query: {query}

        SQL Query:
        """
    # GEMINI CALL.
    client = genai.Client(
        vertexai=True,
        project="pe-rnd",
        location="us-central1",
    )
    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
    )
    resp = client.models.generate_content(model=model, contents=contents, config=generate_content_config)
    if resp.text:
        generated_sql_query = resp.text
        match = re.search(r"```sql\s*(.*?)\s*```", generated_sql_query,
                          re.DOTALL)  # Use re.DOTALL to match across newlines
        if match:
            print(f"match: {match}")
            return match.group(1).strip().replace("\n",
                                                  " ")  # Return the captured group (the SQL query) and strip whitespace
        else:
            return generated_sql_query

    else:
        "No sql query generated."


def process_element(data):
    resp = {"column_name": data['column_name'], "column_data_type": data['column_data_type'],
            "description": data['description']}
    return resp


def process_csv_file(file):
    df = pd.read_csv(file)
    table_list = df['table_name'].unique().tolist()
    data_dict = {}
    for _table in table_list:
        data_dict[_table] = []
        table_df = df[df['table_name'] == _table]
        resp = table_df.apply(process_element, axis=1)
        data_dict[_table].extend(resp.tolist())
    return data_dict


def generate_result(sql_query, db_name):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        # print(f"cursor_description: {cursor.description}")
        cursor_cols = cursor.description
        df_columns = []
        for _cols in cursor_cols:
            df_columns.append(_cols[0])
        resp = cursor.fetchall()
        print(f"SQL QUERY: {sql_query}")
        print("*" * 50)
        print(f"columns: {df_columns}")
        print(f"resp: {resp}")
        print("##" * 50)
        if len(resp) >= 0:
            df = pd.DataFrame(resp, columns=df_columns)
            if 'index' in df.columns:
                df.drop(['index'], axis=1, inplace=True)
            # df = pd.DataFrame(resp)

            file_path = f"data/output_result_{randint(1, 100)}.csv"
            df.to_csv(file_path, index=False)
            data = json.dumps(df.to_dict())
            return True, file_path, data, sql_query
        else:
            return False, "", "", sql_query
    except Exception as e:
        return False, "", "", sql_query


def create_embeddings(metadata):
    """Creates embeddings for table descriptions."""
    embeddings = {}
    for table in metadata:
        table_name = table['table_name']
        description = table['description']
        embedding = model_embedding.encode(description)
        embeddings[table_name] = embedding
    return embeddings


def similarity_search(query: str, embeddings, top_k: int = 8):
    """Finds the most similar tables based on the query."""

    query_embedding = model_embedding.encode(query)
    similarities = {}
    for table_name, embedding in embeddings.items():
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities[table_name] = similarity

    sorted_tables = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    print(f"sorted_tables: {sorted_tables}")
    necessary_tables = [table_name for table_name, _ in sorted_tables[:top_k]]
    return necessary_tables


def create_prompt(query: str, relevant_tables) -> str:
    """Creates the prompt for the Gemini model."""
    table_context = ""
    for table in relevant_tables:
        table_context += f"Table Name: {table['table_name']}\n"
        table_context += f"Description: {table['description']}\n"
        table_context += "Columns:\n"
        for column in table['columns']:
            table_context += f"  - {column['columns_name']}: {column['description']} (Data Type: {column['column_data_type']})\n"
        table_context += "\n"

    prompt = f"""
    You are a SQL expert. Generate a SQL query to answer the question based on the provided table schemas. You have to consider provided table description as key parameter to choose table.
    generate a valid SQL query to answer the question.
    Stick with given schema only. Do not hallucinate the answer. You have to provide sql query only. For providing the sql queries, 
    you can use the table join wherever applicable. Do not use unless it is require. 
    You will be rewarded wih 100 bucks for providing correct answer and penalized very heavily 1000 bucks in case of providing wrong answer.
    You only have to provide the query. You do not have to add any extra content in output response.
    Output type:
    Your output must contains a JSON structure. 
    First key should be success and it's value must be boo.
    True: when sql query generated successfully.
    False: when sql query not generated successfully.
    Second key should be data and it's value must be the LLM provided response.

table schemas:
{table_context}

Question: {query}

SQL Query:
"""
    return prompt


def generate_sql_using_gcp_llm_gemini_flash(prompt):
    import re
    # GEMINI CALL.
    client = genai.Client(
        vertexai=True,
        project=os.getenv("project", ""),
        location=os.getenv("location", ""),
    )
    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
    )
    resp = client.models.generate_content(model=model, contents=contents, config=generate_content_config)
    if resp.text:
        generated_sql_query = resp.text
        print(f"generated_sql_query: {generated_sql_query}")

        # match = re.search(r"```sql\s*(.*?)\s*```", generated_sql_query,
        #                   re.DOTALL)  # Use re.DOTALL to match across newlines
        delimited_parameter = generated_sql_query.split("\n")[0]
        match = re.search(rf'{delimited_parameter}s*(.*?)\s*```', generated_sql_query,
                          re.DOTALL)  # Use re.DOTALL to match across newlines
        if match:
            str_data = match.group(1).strip().replace("\n",
                                                      " ")  # Return the captured group (the SQL query) and strip whitespace
            import json
            dict_data = json.loads(str_data)
            return dict_data
        else:
            return {"success": False, "data": generated_sql_query}

    else:
        return {"success": False, "data": ""}
