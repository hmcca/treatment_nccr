import os

# File paths
CHECKPOINT_FILE = "drug_extraction_checkpoint.csv"
DATA_FILE = "sample_dummy_dataset.csv"
BACKUP_DIR = "checkpoint_backups"
DRUG_SYNONYM_FILE = "cleaned_hemonc_regimen_drug_synonym.csv"
REGIMEN_DRUG_FILE = "/gpfs/wolf2/cades/med128/scratch/uw8/project/rag_test/regimen_drug_data.csv"

# Model settings
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 40
TEMPERATURE = 0.1
TOP_P = 0.95
MAX_TOKENS = 2048

# Processing settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
FUZZY_MATCH_THRESHOLD = 85  # percentage

# Schema definition
DRUG_SCHEMA = '''{
    "type": "object",
    "properties": {
        "drugs": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 0
        },
        "regimens": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 0
        }
    },
    "required": ["drugs", "regimens"]
}'''

# Create backup directory if it doesn't exist
os.makedirs(BACKUP_DIR, exist_ok=True) 