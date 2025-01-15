from datetime import datetime
from decimal import Decimal

def normalize_list(lst):
    """
    Normalizes a list by converting None to empty strings and stripping whitespace from elements.
    
    Args:
        lst (list): The list to normalize.
        
    Returns:
        list: A normalized list with None replaced by empty strings.
    """
    return [str(item).strip().lower() if item is not None else '' for item in lst]

def compare_answers(gt, answer):
    """
    Compares the ground truth (gt) and the model's answer, handling different data types.
    
    Args:
        gt (str): The ground truth answer.
        answer (str): The model's processed answer.
        
    Returns:
        bool: True if the answers match, False otherwise.
    """

    if isinstance(gt, list) and isinstance(answer, list):
        return normalize_list(gt) == normalize_list(answer)
    
    # Convert both to strings for flexible comparison
    gt, answer = str(gt).strip(), str(answer).strip()
    
    # Try datetime comparison
    try:
        gt_dt = datetime.fromisoformat(gt.replace("Z", "+00:00"))
        answer_dt = datetime.fromisoformat(answer.replace("Z", "+00:00"))
        return gt_dt.date() == answer_dt.date()  # Compare only the date part
    except ValueError:
        pass  # Continue to other comparisons if datetime parsing fails
    
    # Try numerical comparison (ignoring small formatting differences)
    try:
        return float(gt) == float(answer)
    except ValueError:
        pass  # Continue if not numerical
    
    # Try boolean comparison
    if gt.lower() in ['true', 'false'] and answer.lower() in ['true', 'false']:
        return gt.lower() == answer.lower()
    
    # Fallback: string comparison (case-insensitive)
    return gt.lower() == answer.lower()