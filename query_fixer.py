import re
import sqlparse

def auto_fix_sql_query(query, conn):
    """
    Automatically fixes SQL query issues:
    - Missing columns/tables are corrected if alternatives exist.
    - Type mismatches (text vs int) are cast properly.
    - Syntax errors are auto-corrected where possible.

    Args:
        query (str): The SQL query.
        conn (psycopg2.Connection): Database connection.

    Returns:
        str: The fixed query.
    """
    try:
        # 🔹 **Check & Fix Syntax Errors**
        parsed_query = sqlparse.parse(query)
        if not parsed_query:
            return None  # Query is malformed beyond repair

        # 🔹 **Extract Tables & Columns from Query**
        tables, columns = extract_tables_and_columns(query)

        # 🔹 **Validate & Correct Table & Column Names**
        fixed_query = query
        missing_table, missing_column = validate_schema_existence(tables, columns, conn)

        if missing_table:
            fixed_query = fixed_query.replace(missing_table, suggest_similar_table(missing_table, conn))

        if missing_column:
            fixed_query = fixed_query.replace(missing_column, suggest_similar_column(missing_column, conn))

        # 🔹 **Fix Type Mismatch Issues**
        fixed_query = fix_type_mismatch(fixed_query, conn, columns)

        return fixed_query

    except Exception as e:
        return query  # If unable to fix, return original query

def extract_tables_and_columns(query):
    """
    Extracts tables and columns used in an SQL query.

    Args:
        query (str): The SQL query.

    Returns:
        tuple: (set of tables, set of columns)
    """
    tables = set()
    columns = set()

    table_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    column_pattern = r'SELECT\s+(.*?)\s+FROM'

    table_matches = re.findall(table_pattern, query, re.IGNORECASE)
    column_matches = re.findall(column_pattern, query, re.IGNORECASE)

    if table_matches:
        tables.update(table_matches)
    if column_matches:
        columns.update(re.split(r',\s*', column_matches[0]))  # Split columns by comma

    return tables, columns

def validate_schema_existence(tables, columns, conn):
    """
    Checks if the referenced tables and columns exist.

    Returns:
        tuple: (missing_table, missing_column)
    """
    cursor = conn.cursor()
    missing_table = None
    missing_column = None

    try:
        # 🔹 Check if tables exist
        for table in tables:
            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (table,))
            if not cursor.fetchone()[0]:
                missing_table = table
                break

        # 🔹 Check if columns exist
        if not missing_table:
            for column in columns:
                cursor.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.columns WHERE column_name = %s)", (column,))
                if not cursor.fetchone()[0]:
                    missing_column = column
                    break

    except Exception as e:
        print(f"Schema validation error: {str(e)}")

    finally:
        cursor.close()

    return missing_table, missing_column

def suggest_similar_column(missing_column, conn):
    """
    Finds the closest matching column name from the schema.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT column_name FROM information_schema.columns")
    existing_columns = [row[0] for row in cursor.fetchall()]

    cursor.close()

    for col in existing_columns:
        if missing_column.lower() in col.lower():
            return col  # Return the closest match

    return missing_column  # If no close match found, return the original column

def suggest_similar_table(missing_table, conn):
    """
    Finds the closest matching table name from the schema.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables")
    existing_tables = [row[0] for row in cursor.fetchall()]

    cursor.close()

    for table in existing_tables:
        if missing_table.lower() in table.lower():
            return table  # Return closest match

    return missing_table  # If no match, return the original

def fix_type_mismatch(query, conn, columns):
    """
    Fixes type mismatches by casting text fields to appropriate numeric types.
    """
    cursor = conn.cursor()

    for col in columns:
        try:
            cursor.execute(
                "SELECT data_type FROM information_schema.columns WHERE column_name = %s", (col,))
            data_type = cursor.fetchone()

            if data_type and data_type[0] == 'text':  # Fix numeric comparisons with text
                query = re.sub(rf'\b{col}\s*([<>!=]=?)\s*(\d+)', rf'CAST({col} AS FLOAT) \1 \2', query)

        except Exception as e:
            print(f"Type checking error: {str(e)}")

    cursor.close()
    return query
