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

def post_process_result(result, expected_type):
    """
    Post-processes the SQL query result to match the expected format.

    Args:
        result (list): The raw result from the SQL query execution.
        expected_type (str): The expected type of the answer ('boolean', 'category', 'number', 'list[category]', 'list[number]').

    Returns:
        str: A processed result in the correct format.
    """
    # Handle empty results
    if not result or result == [None]:
        return "False" if expected_type == "boolean" else "No result"

    # Process based on expected type
    if expected_type == "boolean":
        # A boolean answer is True/False
        value = result[0][0]  # Assume the first element in the first row
        if isinstance(value, str):
            value = value.lower() in ["true", "yes", "y", "1"]
        elif isinstance(value, (int, float)):
            value = bool(value)
        return "True" if value else "False"

    elif expected_type == "number":
        # Convert the first value to a number
        value = result[0][0]
        try:
            return str(round(float(value), 2))  # Round to 2 decimal places
        except (ValueError, TypeError):
            return "Invalid number"

    elif expected_type == "category":
        # Return the first value as a category
        value = result[0][0]
        return str(value).strip()

    elif expected_type == "list[category]":
        # Extract all distinct category values
        values = [row[0] for row in result if row[0] is not None]
        return f"[{', '.join(map(str, values))}]"

    elif expected_type == "list[number]":
        # Extract all distinct numerical values
        try:
            numbers = [float(row[0]) for row in result if row[0] is not None]
            numbers = sorted(set(numbers))  # Remove duplicates and sort
            return f"[{', '.join(map(lambda x: str(round(x, 2)), numbers))}]"
        except ValueError:
            return "Invalid list of numbers"

    # Fallback for unexpected types
    return "Invalid expected type"

