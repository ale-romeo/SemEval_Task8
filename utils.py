from decimal import Decimal
import re
import ast

def clean_column_name(name):
    """
    Cleans column names by removing special characters and emojis.
    """
    cleaned_name = re.sub(r'[^\w\s]', '', name).strip().lower().replace(' ', '_')
    return cleaned_name

def label_answer_type(gt):
    """
    Labels the expected type of the answer based on its format.
    """
    gt = str(gt).strip()

    # Boolean
    if gt.lower() in ['true', 'false', 'y', 'n', 'yes', 'no']:
        return 'boolean'

    # List
    if gt.startswith('[') and gt.endswith(']'):
        try:
            parsed_list = ast.literal_eval(gt)
            if all(isinstance(item, (int, float)) for item in parsed_list):
                return 'list[number]'
            elif all(isinstance(item, str) for item in parsed_list):
                return 'list[category]'
        except (ValueError, SyntaxError):
            pass

    # Number
    try:
        float(gt)
        return 'number'
    except ValueError:
        pass

    # Default to category
    return 'category'

def post_process_result(result, expected_type, schema):
    """
    Post-processes the SQL query result to match the expected answer type.

    Args:
        result (list): Raw query result from SQL execution.
        expected_type (str): Expected format ('boolean', 'number', 'list[number]', 'category', 'list[category]').
        schema (pd.DataFrame): Schema containing column names and types.

    Returns:
        str: Formatted result.
    """
    if not result:
        return "No result"

    # **Identify column types dynamically**
    numeric_columns = [
        col for col in schema.columns 
        if schema.dtypes[col] in ["float64", "int64", "Float64", "Int64"]
    ]
    text_columns = [
        col for col in schema.columns
        if schema.dtypes[col] in ["object", "string", "category"]
    ]

    # ✅ **Boolean Handling**
    if expected_type == 'boolean':
        return "True" if result and result[0][0] else "False"

    # ✅ **Single Number Extraction**
    elif expected_type == 'number':
        try:
            for row in result:
                numeric_values = [float(value) for value in row if isinstance(value, (int, float, Decimal))]
                if numeric_values:
                    return str(numeric_values[0])  # Extract first valid number
        except Exception:
            return "Invalid number"

    # ✅ **List of Numbers (Generalized Selection)**
    elif expected_type == 'list[number]':
        try:
            numerical_values = []
            for row in result:
                row_values = [float(value) for value in row if isinstance(value, (int, float, Decimal))]

                # If two columns exist and first is string, take second (ignore identifiers)
                if len(row) == 2:
                    first, second = row
                    if isinstance(first, str) and first.strip():  # First column is an identifier (string)
                        numerical_values.append(float(second))  # Take second column
                    else:
                        numerical_values.append(float(first))  # Otherwise, take first if numeric

                # If there are multiple numbers in the row, take the second column if it exists
                elif len(row_values) >= 1:
                    numerical_values.append(row_values[-1])  # Pick the most relevant number

            return str(numerical_values) if numerical_values else "Invalid list of numbers"

        except Exception:
            return "Invalid list of numbers"

    # ✅ **Category Extraction**
    elif expected_type == 'category':
        try:
            return str(result[0][0])  # Extract first column value as string
        except IndexError:
            return "Invalid category"

    # ✅ **List of Categories (Handles `None`)**
    elif expected_type == 'list[category]':
        try:
            extracted_categories = []
            for row in result:
                for value in row:
                    if isinstance(value, str) and value.strip():
                        extracted_categories.append(value.strip())

            return str(extracted_categories) if extracted_categories else "Invalid list of categories"
        except Exception:
            return "Invalid list of categories"

    # ❌ **Fallback for Unexpected Cases**
    return "Invalid format"

