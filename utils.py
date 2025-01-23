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
