import re
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

def setup_postgresql():
    """
    Sets up a PostgreSQL database by creating a user and a database.
    """
    try:
        # Connect to the default PostgreSQL database as the superuser
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",  # Default superuser
            password="postgres",  # Set your superuser password
            host="localhost"
        )
        conn.autocommit = True  # Enable autocommit for DDL statements
        cursor = conn.cursor()
        
        # Create a new user
        cursor.execute("CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';")
        
        # Create a new database
        cursor.execute("CREATE DATABASE mydb;")
        
        # Grant privileges to the new user on the new database
        cursor.execute("GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;")
        
        print("PostgreSQL setup completed successfully.")
    
    except Exception as e:
        print(f"Error during PostgreSQL setup: {e}")
    
    finally:
        if conn:
            cursor.close()
            conn.close()

def execute_sql_query(conn, query):
    """
    Executes the given SQL query on the provided PostgreSQL connection.
    
    Args:
        conn (psycopg2.Connection): The PostgreSQL database connection.
        query (str): The SQL query to execute.
    
    Returns:
        list: The result of the query.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()  # Close the connection after execution
        return result
    except Exception as e:
        return f"Error during execution: {str(e)}"

def generate_sql_prompt(df, dataset_id, question):
    """
    Generates a prompt for the Natural SQL model.
    
    Args:
        df (pd.DataFrame): The dataset as a DataFrame.
        dataset_id (str): The ID of the dataset.
        question (str): The question to include in the prompt.
        
    Returns:
        str: A formatted prompt.
    """
    column_types = ', '.join([f"{col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
    prompt = f"""
# Task 
Generate a PostgreSQL query to answer the following question: `{question}`

### PostgreSQL Database Schema 
The query will run on a table named {dataset_id.split('_')[1].lower()  if dataset_id != '001_Forbes' else "billionaires"} with the following schema: 
{column_types}

# PostgreSQL
Here is the PostgreSQL query that answers the question: `{question}`
```sql
"""
    return prompt

def clean_column_name(name):
    """
    Cleans the column name by removing emojis and special characters.
    
    Args:
        name (str): The original column name.
        
    Returns:
        str: The cleaned column name.
    """
    # Remove emojis and special characters (keep only alphanumeric and underscores)
    cleaned_name = re.sub(r'[^\w\s]', '', name).strip().lower()
    return cleaned_name

def load_dataset_into_db(dataset_id):
    """
    Loads a dataset by its ID and stores it in a PostgreSQL database.
    
    Args:
        dataset_id (str): The ID of the dataset to load.
        
    Returns:
        psycopg2.Connection: A connection to the PostgreSQL database.
    """
    # Load the dataset
    df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dataset_id}/all.parquet")
    
    for col in df.columns:
        if df[col].dtype.kind == 'O' and any(isinstance(val, (list, np.ndarray)) for val in df[col].dropna()):
            df[col] = df[col].apply(
                lambda x: ', '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x) if x is not None else ''
            )
            
    df.columns = [clean_column_name(col) for col in df.columns]
    
    df = df.loc[:,~df.columns.duplicated()].copy()
    
    # Create a PostgreSQL connection using SQLAlchemy engine
    engine = create_engine('postgresql://myuser:mypassword@localhost:5432/mydb')
    
    # Write the DataFrame to the PostgreSQL database
    df.to_sql(dataset_id.split('_')[1].lower() if dataset_id != '001_Forbes' else "billionaires", engine, index=False, if_exists="replace")
    
    # Create a connection using psycopg2
    conn = psycopg2.connect(
        dbname="mydb",
        user="myuser",
        password="mypassword",
        host="localhost",
        port="5432"
    )
    
    return conn, df

def preprocess_query_for_postgresql(query, df_columns):
    """
    Preprocesses the generated SQL query to make it compatible with PostgreSQL by quoting case-sensitive column names.
    
    Args:
        query (str): The generated SQL query.
        df_columns (list): List of column names in the DataFrame.
        
    Returns:
        str: The preprocessed SQL query with correctly quoted column names.
    """
    # Create a regex pattern to match column names in the query, ignoring trailing underscores
    pattern = re.compile(r'\b(' + '|'.join(re.escape(col) for col in df_columns) + r')\b(?!")')
    
    # Replace matched column names with properly quoted versions
    preprocessed_query = pattern.sub(lambda match: f'"{match.group(1)}"', query)
    
    return preprocessed_query

def post_process_result(result, expected_type):
    """
    Post-processes the SQL query result to match the expected format.
    
    Args:
        result (list): The raw result from the SQL query execution.
        expected_type (str): The expected type of the ground truth ('boolean', 'value', etc.).
        
    Returns:
        str: A processed result in the correct format.
    """
    if not result:
        return "False" if expected_type == 'boolean' else "No result"
    
    # Assuming result is a single row with a single value for non-boolean cases
    if expected_type == 'boolean':
        if result[0][0] == 'No':
            return "False"
        return "True" if result else "False"
    elif expected_type == 'array':
        first_terms = [str(row[0]) for row in result]
        return f"[{', '.join(first_terms)}]"
    elif expected_type == 'value':
        value = result[0][0]
        return str(value)
