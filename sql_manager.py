import re
import time
import os
import pandas as pd
import psycopg2
import numpy as np
from SemEval_Task8.utils import clean_column_name

def execute_sql_query(conn, query):
    """
    Executes the given SQL query on a PostgreSQL connection and ensures the connection is closed.

    Args:
        conn (psycopg2.Connection): The active database connection.
        query (str): The SQL query to execute.

    Returns:
        list or str: The query result or an error message.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)

        # Handle SELECT vs. non-SELECT queries
        if cursor.description:
            result = cursor.fetchall()  # Fetch results for SELECT queries
        else:
            conn.commit()  # Commit for INSERT/UPDATE queries
            result = "Query executed successfully."

        cursor.close()  # Close the cursor
        conn.close()  # Close the connection

        return result

    except Exception as e:
        conn.close()  # Ensure connection is closed on error
        return f"Error during execution: {str(e)}"

def generate_sql_prompt(schema, dataset_id, question):
    """
    Generates a SQL prompt for Natural SQL model.

    Args:
        df (pd.DataFrame): The dataset schema as a DataFrame.
        dataset_id (str): The dataset identifier.
        question (str): The question.

    Returns:
        str: Formatted SQL query prompt.
    """
    table_name = dataset_id.split('_')[1].lower() if dataset_id != '001_Forbes' else "billionaires"

    # Format column names and data types for readability
    column_types = '\n'.join([f"- {col}: {dtype}" for col, dtype in zip(schema.columns, schema.dtypes)])

    prompt = f"""
        ### Task
        Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

        ### Database Schema
        The query will run on a database with the following schema:
        {schema}

        ### Answer
        Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
        [SQL]
    """
    return prompt

def load_dataset_into_db(dataset_id, engine, retries=5, delay=5, cache_dir="./hf_cache"):
    """
    Loads a dataset into PostgreSQL and ensures proper connection handling.
    Implements retries & caching to guarantee dataset retrieval.

    Args:
        dataset_id (str): The dataset ID.
        engine: SQLAlchemy database engine.
        retries (int): Number of times to retry the download.
        delay (int): Delay between retries.
        cache_dir (str): Directory to store cached datasets.

    Returns:
        psycopg2.Connection: A connection to the PostgreSQL database.
        pd.DataFrame: The loaded DataFrame.
    """
    dataset_path = os.path.join(cache_dir, f"{dataset_id}.parquet")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Try downloading the dataset with retries
    for attempt in range(retries):
        try:
            df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dataset_id}/all.parquet")
            
            # Save a copy in case of future failures
            df.to_parquet(dataset_path)
            break

        except Exception as e:
            print(f"⚠️ Error downloading dataset {dataset_id}: {str(e)}. Retrying in {delay} seconds...")
            time.sleep(delay)
            df = None  # Ensure df remains None if download fails

    # If download failed, try loading from cache
    if df is None and os.path.exists(dataset_path):
        print(f"🔄 Loading cached dataset {dataset_id} from {dataset_path}")
        df = pd.read_parquet(dataset_path)

    # If dataset is still None, create a dummy DataFrame to prevent crashes
    if df is None or df.empty:
        print(f"❌ Failed to load dataset {dataset_id}. Returning an empty DataFrame.")
        df = pd.DataFrame(columns=["error"], data=[["Dataset unavailable"]])

     # 🔹 **Convert Categorical Columns to Strings**
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype(str)

    # 🔹 **Convert Lists/NumPy Arrays to Strings**
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: ', '.join(map(str, x)) if isinstance(x, (np.ndarray, list)) else str(x) if pd.notnull(x) else ''
        )

    # 🔹 **Convert Datetime Columns to Strings**
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Clean and process column names
    cleaned_columns = [clean_column_name(col)[:63] for col in df.columns]
    df.columns = cleaned_columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]  # Remove duplicate columns

    # Define table name
    table_name = dataset_id.split('_')[1].lower() if dataset_id != '001_Forbes' else "billionaires"

    # Write DataFrame to PostgreSQL
    df.to_sql(table_name, engine, index=False, if_exists="replace")

    # Open a new connection after writing the data
    conn = psycopg2.connect(
        dbname="mydb",
        user="myuser",
        password="mypassword",
        host="localhost",
        connect_timeout=10
    )

    return conn, df

def preprocess_query_for_postgresql(query, df_columns):
    """
    Preprocesses the SQL query for PostgreSQL by escaping case-sensitive column names.

    Args:
        query (str): The generated SQL query.
        df_columns (list): List of table column names.

    Returns:
        str: Preprocessed SQL query with escaped column names.
    """
    try:
        # Ensure column names are formatted correctly for PostgreSQL
        pattern = re.compile(r'\b(' + '|'.join(re.escape(col) for col in df_columns) + r')\b(?!")')
        preprocessed_query = pattern.sub(lambda match: f'"{match.group(1)}"', query)

        return preprocessed_query
    except Exception as e:
        print(f"Error preprocessing query: {str(e)}")
        return query

