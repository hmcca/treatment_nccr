import os
import time
import pandas as pd
import json
from datetime import datetime
import logging
from config import (
    DATA_FILE, BATCH_SIZE, MAX_RETRIES, RETRY_DELAY,
    CHECKPOINT_FILE, DRUG_SYNONYM_FILE, REGIMEN_DRUG_FILE
)
from model_utils import set_hf_env, get_llm, get_sampling_params, get_generator
from data_utils import load_checkpoint, save_checkpoint, process_batch
from metrics import compute_metrics, compute_regimen_metrics, print_avg_metrics, normalize_drug_list
from drug_normalizer import DrugNormalizer, RegimenDrugMapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_with_retry(batch, generator, sampling_params, max_retries=MAX_RETRIES):
    """Process a batch with retry logic."""
    for attempt in range(max_retries):
        try:
            return process_batch(batch, generator, sampling_params)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to process batch after {max_retries} attempts: {str(e)[:200]}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)


def normalize_drugs_in_df(df: pd.DataFrame, normalizer: DrugNormalizer) -> pd.DataFrame:
    """Normalize drug names in the DataFrame."""
    df = df.copy()
    
    # Normalize extracted_drugs
    df['normalized_drugs'] = df['extracted_drugs'].apply(normalizer.normalize_drugs)
    
    # Log some sample normalizations for debugging
    logger.info("\nSample drug normalizations:")
    for _, row in df.head(3).iterrows():
        logger.info(f"Original: {row['extracted_drugs']}")
        logger.info(f"Normalized: {row['normalized_drugs']}")
        logger.info("---")
    
    # Recalculate metrics with normalized drugs
    df[["normalized_precision", "normalized_recall", "normalized_f1", 
        "normalized_missing_drugs", "normalized_hallucinated_drugs"]] = \
        df.apply(lambda row: compute_metrics(row, use_normalized=True), axis=1)
    
    return df


def run_pipeline():
    start_time = datetime.now()
    logger.info(f"Starting pipeline at {start_time.isoformat()}")
    
    # Load existing checkpoint
    df = load_checkpoint()
    logger.info(f"Loaded checkpoint with {len(df)} processed records")
    
    # Load and prepare raw data
    raw_data = pd.read_csv(DATA_FILE)
    raw_data["unique_key"] = (
        raw_data["patient_id_number"].astype(str) + "_" +
        raw_data["tumor_record_number"].astype(str) + "_" +
        raw_data["admission_id"].astype(str)
    )
    
    # Normalize unique_drugs to ensure consistent case
    raw_data["unique_drugs"] = raw_data["unique_drugs"].apply(
        lambda x: normalize_drug_list(x.split(", ")) if isinstance(x, str) else []
    )
    
    # Normalize regimens to ensure consistent case
    raw_data["regimens"] = raw_data["regimens"].apply(
        lambda x: normalize_drug_list(x.split(", ")) if isinstance(x, str) else []
    )
    
    # Find unprocessed records
    processed_keys = set(df["unique_key"]) if not df.empty else set()
    todo = raw_data[~raw_data["unique_key"].isin(processed_keys)]
    
    if todo.empty:
        logger.info("All data processed")
        print_avg_metrics(df)
        return df
    
    total_records = len(todo)
    logger.info(f"Processing {total_records} new records...")
    
    # Model setup
    llm = get_llm()
    sampling_params = get_sampling_params()
    generator = get_generator(llm)
    
    # Initialize drug normalizer and regimen mapper
    normalizer = DrugNormalizer(DRUG_SYNONYM_FILE)
    regimen_mapper = RegimenDrugMapper(REGIMEN_DRUG_FILE)
    
    # Process in batches
    for i in range(0, total_records, BATCH_SIZE):
        batch_start = datetime.now()
        batch = todo.iloc[i:i+BATCH_SIZE].copy()
        batch_size = len(batch)
        
        logger.info(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(total_records//BATCH_SIZE)+1}")
        logger.info(f"Batch size: {batch_size} records")
        
        try:
            # Process batch with retry logic
            batch_results = process_with_retry(batch, generator, sampling_params)
            
            # Merge results
            merged = pd.merge(
                batch,
                batch_results,
                on=["unique_key", "text_concat"],
                how="left"
            )
            
            # Calculate metrics
            if not merged.empty:
                # Calculate drug metrics
                merged[["precision", "recall", "f1", "missing_drugs", "hallucinated_drugs"]] = \
                    merged.apply(compute_metrics, axis=1)
                
                # Calculate regimen metrics
                merged[["precision_regimen", "recall_regimen", "f1_regimen", 
                       "missing_regimens", "hallucinated_regimens"]] = \
                    merged.apply(compute_regimen_metrics, axis=1)
                
                # Normalize drugs and calculate normalized metrics
                merged = normalize_drugs_in_df(merged, normalizer)
                
                # Calculate combined drugs
                merged['combined_drugs'] = merged.apply(
                    lambda row: regimen_mapper.get_combined_drugs(
                        row['extracted_drugs'],
                        row['extracted_regimens']
                    ),
                    axis=1
                )
                
                # Create mapped regimen drugs in JSON format
                merged['mapped_regimen_drugs'] = merged.apply(
                    lambda row: json.dumps({
                        regimen: regimen_mapper.get_drugs_from_regimen(regimen)
                        for regimen in row['extracted_regimens']
                        if regimen
                    }, indent=2),
                    axis=1
                )
            
            # Update and save incrementally
            df = pd.concat([df, merged], ignore_index=True)
            save_checkpoint(df)
            
            # Calculate and display batch statistics
            batch_end = datetime.now()
            batch_duration = (batch_end - batch_start).total_seconds()
            records_per_second = batch_size / batch_duration if batch_duration > 0 else 0
            
            logger.info(f"Batch completed in {batch_duration:.2f} seconds")
            logger.info(f"Processing speed: {records_per_second:.2f} records/second")
            
            # Calculate and display overall progress
            processed = len(df)
            remaining = total_records - (processed - len(processed_keys))
            progress = (processed - len(processed_keys)) / total_records * 100
            logger.info(f"Progress: {progress:.1f}% ({processed - len(processed_keys)}/{total_records} records)")
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)[:200]}")
            logger.info("Saving current progress before exiting...")
            save_checkpoint(df)
            raise
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\nPipeline completed in {duration:.2f} seconds")
    
    # Print metrics comparison
    print("\nMetrics before normalization:")
    print_avg_metrics(df, use_normalized=False)
    print("\nMetrics after normalization:")
    print_avg_metrics(df, use_normalized=True)
    
    return df


if __name__ == "__main__":
    try:
        # Activate conda env and set env vars
        set_hf_env()
        final_df = run_pipeline()
        print("\nPipeline completed successfully. Sample output:")
        if not final_df.empty:
            print(final_df[["unique_key", "text_concat", "extracted_drugs", "normalized_drugs"]].head(2).to_string(index=False))
    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)[:200]}")
        raise 