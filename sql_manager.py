import re
import time
import os
import pandas as pd
import psycopg2
import numpy as np
from sqlalchemy import BigInteger
from SemEval_Task8.utils import clean_column_name
from sqlalchemy.types import Integer, Float, Boolean, String, DateTime

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
        result = cursor.fetchall()  # Fetch results for SELECT queries

        cursor.close()  # Close the cursor
        conn.close()  # Close the connection

        return result

    except Exception as e:
        conn.close()  # Ensure connection is closed on error
        return f"Error during execution: {str(e)}"

def generate_sql_prompt(schema, dataset_id, question, predicted_answer_type):
    """
    Generates a SQL prompt for the model with detailed aggregation handling.
    """
    table_name = dataset_id.split('_')[1].lower() if dataset_id != '001_Forbes' else "billionaires"

    column_types = '\n'.join([f"- {col}: {dtype}" for col, dtype in zip(schema.columns, schema.dtypes)])

    system_prompt = "SYSTEM: You are an expert in writing optimized SQL queries."
    context = (
        "CONTEXT: Use PostgreSQL syntax. The query must be well-formed, handle NULL values, "
        "and ensure proper aggregations. If an `ORDER BY` clause is used, all non-aggregated columns "
        "in the `SELECT` clause must appear in the `GROUP BY` clause."
    )

    prompt = f"""
    {system_prompt}
    {context}

    #### Important Instructions:
    - The query must use only one table.
    - DO NOT use JOINs.
    - Always assign the alias `X` to the table.
    - If your query includes `ORDER BY`, ensure all non-aggregated columns in `SELECT` appear in `GROUP BY`.

    ### Example
    Handling Aggregations
    ```
    SELECT X.category, SUM(CAST(X.sales AS FLOAT)) AS total_sales
    FROM sales AS X
    GROUP BY X.category
    ORDER BY total_sales DESC;
    ```
    
    ### Task
    Generate a SQL query to answer [QUESTION]{question}[/QUESTION].
    The answer to the question is expected to be of type: {predicted_answer_type}.

    ### Table Schema
    The query will run on ONE SINGLE table named `{table_name}` with the following schema:
    {column_types}

    ### Table
    This is the table you will be querying:
    {schema.head(5)}

    ### Answer
    Given the table schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]:
    [SQL]
    """
    
    return prompt

def map_dtype_to_sqlalchemy(dtype):
    """
    Maps Pandas dtypes to SQLAlchemy-compatible types.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return BigInteger if dtype == "Int64" else Integer
    elif pd.api.types.is_float_dtype(dtype):
        return Float
    elif pd.api.types.is_bool_dtype(dtype):
        return Boolean
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return DateTime
    else:
        return String  # Default fallback

def load_dataset_into_db(dataset_id, engine, retries=5, delay=5, cache_dir="./hf_cache"):
    """
    Loads a dataset into PostgreSQL while ensuring proper type conversion.
    Implements retries & caching to guarantee dataset retrieval.

    Args:
        dataset_id (str): The dataset ID.
        engine: SQLAlchemy database engine.
        retries (int): Number of times to retry the download.
        delay (int): Delay between retries.
        cache_dir (str): Directory to store cached datasets.

    Returns:
        psycopg2.Connection: A connection to the PostgreSQL database.
        pd.DataFrame: The loaded DataFrame with correct column types.
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

    # ✅ Restore Correct Data Types
    for col in df.columns:
        # Convert `category` columns back to `category`
        if df[col].dtype.name == "category":
            df[col] = df[col].astype("category")

        # Preserve Numeric Columns (Avoid Converting to `object`)
        elif df[col].dtype.name in ["uint16", "uint32", "float64", "int64"]:
            if df[col].max() > 2_147_483_647:  # Exceeds standard INT range
                df[col] = df[col].astype('Int64')  # Convert to `BIGINT`
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Preserve Boolean Columns
        elif df[col].dtype.name == "bool":
            df[col] = df[col].astype(bool)

        # Convert `datetime` columns to actual datetime64 format
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

        # Convert Lists/NumPy Arrays to Strings for compatibility
        elif any(isinstance(val, (np.ndarray, list)) for val in df[col].dropna()):
            df[col] = df[col].apply(
                lambda x: ', '.join(map(str, x)) if isinstance(x, (np.ndarray, list)) else str(x) if pd.notnull(x) else ''
            )

    # Clean and process column names
    cleaned_columns = [clean_column_name(col)[:63] for col in df.columns]
    df.columns = cleaned_columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]  # Remove duplicate columns

    # Define table name
    table_name = dataset_id.split('_')[1].lower() if dataset_id != '001_Forbes' else "billionaires"

    # ✅ **Explicitly map column types**
    dtype_mapping = {col: map_dtype_to_sqlalchemy(df[col].dtype) for col in df.columns}

    # Write DataFrame to PostgreSQL with explicit dtypes
    df.to_sql(table_name, engine, index=False, if_exists="replace", dtype=dtype_mapping)

    # Open a new connection after writing the data
    conn = psycopg2.connect(
        dbname="mydb",
        user="myuser",
        password="mypassword",
        host="localhost",
        connect_timeout=10
    )

    return conn, df

def preprocess_query_for_postgresql(query, schema):
    """
    Preprocesses SQL queries by:
    - Ensuring `GROUP BY` includes all non-aggregated columns.
    - Fixing SUM, AVG, COUNT() on text fields by casting to FLOAT.
    - Fixing WHERE conditions for numeric and boolean values.
    - Handling NULL values with COALESCE().
    """
    try:
        if isinstance(schema, pd.DataFrame):
            df_columns = schema.columns.tolist()  
            column_types = schema.dtypes  
        else:
            raise ValueError("Schema must be a Pandas DataFrame.")

        # Identify column types
        numeric_columns = [col for col in df_columns if column_types[col] in ["float64", "int64", "Float64", "Int64"]]
        boolean_columns = [col for col in df_columns if column_types[col] in ["bool", "boolean"]]
        text_columns = [col for col in df_columns if column_types[col] in ["object", "string", "category"]]

        # ✅ Ensure ORDER BY numeric casting
        for col in numeric_columns:
            query = re.sub(
                rf'ORDER BY (\w+)\.{col}',
                f'ORDER BY CAST({col} AS FLOAT)',
                query,
                flags=re.IGNORECASE
            )

        # ✅ Fix SUM, AVG, COUNT on text fields by casting to FLOAT
        for col in numeric_columns:
            query = re.sub(
                rf'(SUM|AVG|COUNT)\(\s*{col}\s*\)',
                r'\1(CAST({col} AS FLOAT))',
                query,
                flags=re.IGNORECASE
            )

        # ✅ Fix comparisons (TEXT vs. NUMBER)
        for col in numeric_columns:
            query = re.sub(
                rf'WHERE\s+{col}\s*([=><!]+)\s*(\d+)',
                rf'WHERE CAST({col} AS FLOAT) \1 \2',
                query,
                flags=re.IGNORECASE
            )

        # ✅ Fix boolean comparisons (TEXT vs. BOOLEAN)
        for col in boolean_columns:
            query = re.sub(
                rf'WHERE\s+{col}\s*=\s*(TRUE|FALSE)',
                rf'WHERE CAST({col} AS BOOLEAN) = \1',
                query,
                flags=re.IGNORECASE
            )

        # ✅ Fix missing `GROUP BY` columns
        if "GROUP BY" in query:
            select_columns = re.findall(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)
            if select_columns:
                select_columns = select_columns[0].split(",")
                select_columns = [col.strip().replace("AS", "").split()[0] for col in select_columns]

                group_by_columns = re.findall(r'GROUP BY\s+(.*?)(ORDER BY|LIMIT|$)', query, re.IGNORECASE)
                if group_by_columns:
                    group_by_columns = group_by_columns[0][0].split(",")
                    group_by_columns = [col.strip() for col in group_by_columns]
                else:
                    group_by_columns = []

                missing_group_by = [col for col in select_columns if col not in group_by_columns and col in df_columns]
                if missing_group_by:
                    group_by_clause = f"GROUP BY {', '.join(group_by_columns + missing_group_by)}"
                    query = re.sub(r'GROUP BY\s+(.*?)(ORDER BY|LIMIT|$)', group_by_clause + r' \2', query, flags=re.IGNORECASE)

        # ✅ Fix `MEDIAN()` Calls
        query = re.sub(
            r'MEDIAN\(\s*([\w\.]+)\s*\)',
            r'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY \1)',
            query,
            flags=re.IGNORECASE
        )

        return query

    except Exception as e:
        print(f"⚠️ Error preprocessing query: {str(e)}")
        return query

