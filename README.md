# NCCR Treatment Normalization Pipeline

This repository contains a comprehensive pipeline for extracting and normalizing drug and regimen information from clinical text data. The system uses large language models (LLMs) to extract drugs and regimens from clinical notes and provides sophisticated normalization and mapping capabilities.

## Features

- **Drug Extraction**: Extracts drug names from clinical text using LLMs
- **Regimen Extraction**: Identifies treatment regimens from clinical text
- **Drug Normalization**: Normalizes drug names using synonym mapping
- **Regimen-to-Drug Mapping**: Maps treatment regimens to their constituent drugs
- **Combined Drug Analysis**: Creates comprehensive drug lists from both direct extraction and regimen mapping
- **Metrics Computation**: Calculates precision, recall, and F1 scores for both drugs and regimens
- **Fuzzy Matching**: Uses advanced fuzzy matching for regimen identification

## Project Structure

```
nccr_normaliztion/
├── main.py                 # Main pipeline execution script
├── config.py              # Configuration settings
├── data_utils.py          # Data loading and processing utilities
├── model_utils.py         # LLM model setup and utilities
├── metrics.py             # Metrics computation functions
├── drug_normalizer.py     # Drug normalization and regimen mapping
├── test_mapping.py        # Test script for regimen-drug mapping
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Key Components

### 1. Drug Normalization (`drug_normalizer.py`)
- `DrugNormalizer`: Normalizes drug names using synonym mapping
- `RegimenDrugMapper`: Maps treatment regimens to constituent drugs using fuzzy matching

### 2. Metrics Computation (`metrics.py`)
- `compute_metrics()`: Calculates drug extraction metrics
- `compute_regimen_metrics()`: Calculates regimen extraction metrics
- `print_avg_metrics()`: Displays comprehensive metrics summary

### 3. Pipeline (`main.py`)
- Batch processing with retry logic
- Incremental checkpointing
- Progress tracking and logging
- Combined drug analysis

## Configuration

The system is configured through `config.py`:

- **Model Settings**: LLM model name, temperature, batch size
- **File Paths**: Data files, checkpoint files, mapping files
- **Processing Parameters**: Retry logic, delays, thresholds

## Usage

### Prerequisites

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables for Hugging Face:
```bash
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/hf_home"
```

3. Ensure data files are in place:
- `sample_dummy_dataset.csv`: Input clinical data
- `cleaned_hemonc_regimen_drug_synonym.csv`: Drug synonym mapping
- `regimen_drug_mapping.csv`: Regimen to drug mapping

### Running the Pipeline

```bash
python main.py
```

The pipeline will:
1. Load existing checkpoints (if any)
2. Process new data in batches
3. Extract drugs and regimens using LLMs
4. Normalize drug names
5. Map regimens to constituent drugs
6. Calculate comprehensive metrics
7. Save results to checkpoint files

## Output Columns

The pipeline generates several output columns:

- `extracted_drugs`: Directly extracted drug names
- `extracted_regimens`: Extracted treatment regimens
- `normalized_drugs`: Normalized drug names
- `combined_drugs`: Unique drugs from both extraction and regimen mapping
- `regimen_drugs_only`: Drugs derived only from regimen mapping (flat list)
- `precision/recall/f1`: Drug extraction metrics
- `precision_regimen/recall_regimen/f1_regimen`: Regimen extraction metrics
- `missing_drugs/hallucinated_drugs`: Drug extraction analysis
- `missing_regimens/hallucinated_regimens`: Regimen extraction analysis

## Metrics

The system provides comprehensive metrics for both drug and regimen extraction:

### Drug Metrics
- **Precision**: Accuracy of extracted drugs
- **Recall**: Completeness of drug extraction
- **F1 Score**: Harmonic mean of precision and recall
- **Missing Drugs**: Drugs in ground truth but not extracted
- **Hallucinated Drugs**: Drugs extracted but not in ground truth

### Regimen Metrics
- **Precision**: Accuracy of extracted regimens
- **Recall**: Completeness of regimen extraction
- **F1 Score**: Harmonic mean of precision and recall
- **Missing Regimens**: Regimens in ground truth but not extracted
- **Hallucinated Regimens**: Regimens extracted but not in ground truth

## Advanced Features

### Fuzzy Matching
The regimen mapping uses sophisticated fuzzy matching with:
- Multiple matching algorithms (ratio, partial_ratio, token_sort_ratio)
- Configurable similarity thresholds
- Name normalization (removing common variations)

### Batch Processing
- Configurable batch sizes for memory efficiency
- Retry logic for robustness
- Progress tracking and logging
- Incremental checkpointing

### Data Persistence
- Automatic checkpoint creation
- Backup management
- Resume capability for interrupted runs

## Testing

Run the test script to verify regimen-drug mapping functionality:

```bash
python test_mapping.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nccr_treatment_normalization,
  title={NCCR Treatment Normalization Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/hmcca/treatment_nccr}
}
``` 