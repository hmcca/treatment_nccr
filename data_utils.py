import os
import pandas as pd
import json
import shutil
from datetime import datetime
from config import CHECKPOINT_FILE, DATA_FILE, BACKUP_DIR
from model_utils import format_prompt

def safe_json_loads(x):
    if pd.isna(x) or x.strip() in ['', '{}', '[]']:
        return []
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return []

def create_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

def get_backup_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(BACKUP_DIR, f"checkpoint_backup_{timestamp}.csv")

def atomic_write(df, filename):
    """Write DataFrame to a temporary file and then rename it to the target file."""
    temp_file = f"{filename}.tmp"
    try:
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, filename)
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def process_batch(batch_df, generator, sampling_params):
    prompts = [format_prompt(text) for text in batch_df["text_concat"]]
    results = []
    try:
        # Generate responses for all prompts in the batch
        responses = generator(prompts, sampling_params=sampling_params)
        
        # Process all responses in parallel
        for response, (_, row) in zip(responses, batch_df.iterrows()):
            try:
                if isinstance(response, str):
                    data = json.loads(response)
                else:
                    data = response
                drugs = data.get("drugs", [])
                regimens = data.get("regimens", [])
                results.append({
                    "unique_key": row["unique_key"],
                    "text_concat": row["text_concat"],
                    "json_extraction": json.dumps(data, ensure_ascii=False),
                    "extracted_drugs": drugs,
                    "extracted_regimens": regimens,
                    "processing_timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error processing response for key {row['unique_key']}: {str(e)[:200]}")
                results.append({
                    "unique_key": row["unique_key"],
                    "text_concat": row["text_concat"],
                    "json_extraction": "{}",
                    "extracted_drugs": [],
                    "extracted_regimens": [],
                    "processing_timestamp": datetime.now().isoformat(),
                    "error": str(e)[:200]
                })
    except Exception as e:
        print(f"Batch failed: {str(e)[:200]}")
        for _, row in batch_df.iterrows():
            results.append({
                "unique_key": row["unique_key"],
                "text_concat": row["text_concat"],
                "json_extraction": "{}",
                "extracted_drugs": [],
                "extracted_regimens": [],
                "processing_timestamp": datetime.now().isoformat(),
                "error": str(e)[:200]
            })
    return pd.DataFrame(results)

def load_checkpoint():
    create_backup_dir()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df = pd.read_csv(
                CHECKPOINT_FILE,
                converters={
                    'unique_drugs': safe_json_loads,
                    'extracted_drugs': safe_json_loads,
                    'json_extraction': safe_json_loads
                }
            )
            required_columns = ['unique_key', 'text_concat', 'extracted_drugs']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Corrupted checkpoint - missing required columns")
            return df
        except Exception as e:
            print(f"Checkpoint reset due to error: {str(e)[:200]}")
            backup_file = get_backup_filename()
            try:
                shutil.copy2(CHECKPOINT_FILE, backup_file)
                print(f"Created backup at {backup_file}")
            except Exception as backup_error:
                print(f"Failed to create backup: {str(backup_error)[:200]}")
            os.rename(CHECKPOINT_FILE, f"{CHECKPOINT_FILE}.corrupted")
            return pd.DataFrame()
    return pd.DataFrame()

def save_checkpoint(df):
    """Save checkpoint with atomic write and backup."""
    try:
        # Create a backup before saving
        if os.path.exists(CHECKPOINT_FILE):
            backup_file = get_backup_filename()
            shutil.copy2(CHECKPOINT_FILE, backup_file)
        
        # Perform atomic write
        atomic_write(df, CHECKPOINT_FILE)
        
        # Clean up old backups (keep last 5)
        backup_files = sorted([f for f in os.listdir(BACKUP_DIR) if f.startswith("checkpoint_backup_")])
        if len(backup_files) > 5:
            for old_backup in backup_files[:-5]:
                os.remove(os.path.join(BACKUP_DIR, old_backup))
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)[:200]}")
        raise 