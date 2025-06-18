import pandas as pd
from typing import Dict, Any, List
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_drug_list(drugs: List[str]) -> List[str]:
    """Normalize a list of drug names to lowercase and strip whitespace."""
    if not drugs:
        return []
    return [str(drug).strip().lower() for drug in drugs if drug]

def compute_metrics(row: pd.Series, use_normalized: bool = False) -> pd.Series:
    """
    Compute precision, recall, and F1 score for drug extraction.
    
    Args:
        row: DataFrame row containing extracted and ground truth drugs
        use_normalized: Whether to use normalized drug names for comparison
        
    Returns:
        Series containing precision, recall, F1, missing drugs, and hallucinated drugs
    """
    # Get the appropriate drug lists based on whether we're using normalized names
    extracted_drugs = row['normalized_drugs'] if use_normalized else row['extracted_drugs']
    ground_truth = row['unique_drugs']
    
    # Normalize both lists to lowercase for comparison
    extracted_set = set(normalize_drug_list(extracted_drugs))
    ground_truth_set = set(normalize_drug_list(ground_truth))
    
    # Log the sets being compared for debugging
    logger.debug(f"Extracted drugs: {extracted_set}")
    logger.debug(f"Ground truth drugs: {ground_truth_set}")
    
    # Calculate metrics
    true_positives = len(extracted_set.intersection(ground_truth_set))
    false_positives = len(extracted_set - ground_truth_set)
    false_negatives = len(ground_truth_set - extracted_set)
    
    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Get missing and hallucinated drugs (preserve original case from input)
    missing_drugs = [drug for drug in ground_truth if drug.lower().strip() in (ground_truth_set - extracted_set)]
    hallucinated_drugs = [drug for drug in extracted_drugs if drug.lower().strip() in (extracted_set - ground_truth_set)]
    
    # Log metrics for debugging
    logger.debug(f"True positives: {true_positives}")
    logger.debug(f"False positives: {false_positives}")
    logger.debug(f"False negatives: {false_negatives}")
    logger.debug(f"Precision: {precision:.3f}")
    logger.debug(f"Recall: {recall:.3f}")
    logger.debug(f"F1: {f1:.3f}")
    
    return pd.Series({
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'missing_drugs': missing_drugs,
        'hallucinated_drugs': hallucinated_drugs
    })

def compute_regimen_metrics(row: pd.Series) -> pd.Series:
    """
    Compute precision, recall, and F1 score for regimen extraction.
    
    Args:
        row: DataFrame row containing extracted and ground truth regimens
        
    Returns:
        Series containing precision, recall, F1, missing regimens, and hallucinated regimens
    """
    # Get the regimen lists
    extracted_regimens = row['extracted_regimens']
    ground_truth = row['regimens']
    
    # Convert to sets for comparison
    extracted_set = set(str(reg).strip().lower() for reg in extracted_regimens if reg)
    ground_truth_set = set(str(reg).strip().lower() for reg in ground_truth if reg)
    
    # Calculate metrics
    true_positives = len(extracted_set.intersection(ground_truth_set))
    false_positives = len(extracted_set - ground_truth_set)
    false_negatives = len(ground_truth_set - extracted_set)
    
    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Get missing and hallucinated regimens (preserve original case from input)
    missing_regimens = [reg for reg in ground_truth if str(reg).lower().strip() in (ground_truth_set - extracted_set)]
    hallucinated_regimens = [reg for reg in extracted_regimens if str(reg).lower().strip() in (extracted_set - ground_truth_set)]
    
    return pd.Series({
        'precision_regimen': precision,
        'recall_regimen': recall,
        'f1_regimen': f1,
        'missing_regimens': missing_regimens,
        'hallucinated_regimens': hallucinated_regimens
    })

def print_avg_metrics(df: pd.DataFrame, use_normalized: bool = False) -> None:
    """
    Print average metrics for the entire dataset.
    
    Args:
        df: DataFrame containing the metrics
        use_normalized: Whether to use normalized metrics
    """
    prefix = 'normalized_' if use_normalized else ''
    
    # Calculate drug metrics averages
    avg_precision = df[f'{prefix}precision'].mean()
    avg_recall = df[f'{prefix}recall'].mean()
    avg_f1 = df[f'{prefix}f1'].mean()
    
    # Count missing and hallucinated drugs
    missing_drugs = [drug for drugs in df[f'{prefix}missing_drugs'] for drug in drugs]
    hallucinated_drugs = [drug for drugs in df[f'{prefix}hallucinated_drugs'] for drug in drugs]
    
    missing_counts = Counter(missing_drugs)
    hallucinated_counts = Counter(hallucinated_drugs)
    
    # Print drug metrics
    print(f"\nDrug Metrics:")
    print(f"Precision: {avg_precision:.3f}")
    print(f"Recall: {avg_recall:.3f}")
    print(f"F1 Score: {avg_f1:.3f}")
    
    print("\nTop 5 Most Common Missing Drugs:")
    for drug, count in missing_counts.most_common(5):
        print(f"{drug}: {count} times")
    
    print("\nTop 5 Most Common Hallucinated Drugs:")
    for drug, count in hallucinated_counts.most_common(5):
        print(f"{drug}: {count} times")
    
    # Calculate regimen metrics averages
    avg_precision_regimen = df['precision_regimen'].mean()
    avg_recall_regimen = df['recall_regimen'].mean()
    avg_f1_regimen = df['f1_regimen'].mean()
    
    # Count missing and hallucinated regimens
    missing_regimens = [reg for regs in df['missing_regimens'] for reg in regs]
    hallucinated_regimens = [reg for regs in df['hallucinated_regimens'] for reg in regs]
    
    missing_regimen_counts = Counter(missing_regimens)
    hallucinated_regimen_counts = Counter(hallucinated_regimens)
    
    # Print regimen metrics
    print(f"\nRegimen Metrics:")
    print(f"Precision: {avg_precision_regimen:.3f}")
    print(f"Recall: {avg_recall_regimen:.3f}")
    print(f"F1 Score: {avg_f1_regimen:.3f}")
    
    print("\nTop 5 Most Common Missing Regimens:")
    for reg, count in missing_regimen_counts.most_common(5):
        print(f"{reg}: {count} times")
    
    print("\nTop 5 Most Common Hallucinated Regimens:")
    for reg, count in hallucinated_regimen_counts.most_common(5):
        print(f"{reg}: {count} times")
        
    # Log some sample comparisons for debugging
    logger.info("\nSample comparisons:")
    for _, row in df.head(3).iterrows():
        extracted_drugs = row['normalized_drugs'] if use_normalized else row['extracted_drugs']
        truth_drugs = row['unique_drugs']
        extracted_regimens = row['extracted_regimens']
        truth_regimens = row['regimens']
        
        logger.info(f"Drugs - Extracted: {extracted_drugs}")
        logger.info(f"Drugs - Truth: {truth_drugs}")
        logger.info(f"Drugs - Metrics: P={row[f'{prefix}precision']:.3f}, R={row[f'{prefix}recall']:.3f}, F1={row[f'{prefix}f1']:.3f}")
        
        logger.info(f"Regimens - Extracted: {extracted_regimens}")
        logger.info(f"Regimens - Truth: {truth_regimens}")
        logger.info(f"Regimens - Metrics: P={row['precision_regimen']:.3f}, R={row['recall_regimen']:.3f}, F1={row['f1_regimen']:.3f}")
        logger.info("---") 