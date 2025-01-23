import re
import pandas as pd
import psycopg2

def execute_sql_query(conn, query):
    """
    Executes the given SQL query on the provided PostgreSQL connection.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()  # Close the cursor
        return result
    except Exception as e:
        return f"Error during execution: {str(e)}"

def generate_sql_prompt(df, dataset_id, question):
    """
    Generates a SQL prompt for a given dataset and question.
    """
    column_types = ', '.join([f"{col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
    prompt = f"""
# Task 
Generate a PostgreSQL query to answer the following question: `{question}`

### PostgreSQL Database Schema 
The query will run on a table named {dataset_id.split('_')[1].lower() if dataset_id != '001_Forbes' else "billionaires"} with the following schema: 
{column_types}

# PostgreSQL
Here is the PostgreSQL query that answers the question: `{question}`:
```sql
"""
    return prompt

def load_dataset_into_db(dataset_id, engine):
    """
    Loads a dataset into a PostgreSQL database.
    """
    df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dataset_id}/all.parquet")

    # Process column names and remove duplicates
    from SemEval_Task8.utils import clean_column_name
    cleaned_columns = [clean_column_name(col)[:63] for col in df.columns]
    df.columns = cleaned_columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Write the DataFrame to PostgreSQL
    table_name = dataset_id.split('_')[1].lower() if dataset_id != '001_Forbes' else "billionaires"
    df.to_sql(table_name, engine, index=False, if_exists="replace")

    conn = psycopg2.connect(
        dbname="mydb",
        user="myuser",
        password="mypassword",
        host="localhost"
    )
    return conn, df

def preprocess_query_for_postgresql(query, df_columns):
    """
    Preprocesses the SQL query for PostgreSQL by quoting case-sensitive column names.
    """
    pattern = re.compile(r'\b(' + '|'.join(re.escape(col) for col in df_columns) + r')\b(?!")')
    preprocessed_query = pattern.sub(lambda match: f'"{match.group(1)}"', query)
    return preprocessed_query
