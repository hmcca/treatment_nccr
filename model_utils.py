import os
from outlines import models, generate
from vllm import LLM, SamplingParams
from config import MODEL_NAME, DRUG_SCHEMA

def set_hf_env():
    os.environ["HF_HOME"] = "/gpfs/wolf2/cades/med128/scratch/uw8/huggingface_cache"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_EVALUATE_OFFLINE"] = "1"

def get_llm():
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=True,
        trust_remote_code=False
    )
    return llm

def get_sampling_params():
    return SamplingParams(
        temperature=0.4,
        top_k=200,
        top_p=0.6,
        repetition_penalty=1.1,
        max_tokens=20000,
        seed=30
    )

def get_generator(llm):
    model = models.VLLM(llm)
    generator = generate.json(
        model,
        DRUG_SCHEMA,
        whitespace_pattern=r"[\s]*",
    )
    return generator

def format_prompt(text: str) -> str:
    return f"""Extract and normalize both individual drug names and chemotherapy regimen names from the clinical text in JSON format.
Rules:
1. Normalize drug names to **generic forms**:
   - Convert brand names to generics (e.g., Oncovin → vincristine)
   - Expand abbreviations (e.g., MTX → methotrexate)
   - Correct misspellings (e.g., Methotrxate → methotrexate)
   - If generic equivalent is unknown, include the raw name in lowercase
2. Regimen names should be normalized to lowercase (e.g., R-CHOP → r-chop)
3. Only include drugs explicitly mentioned — do **not** expand regimens into their component drugs
4. Always include both `"drugs"` and `"regimens"` fields in the output
5. Use lowercase only for all entries
Examples:
Input: Started R-CHOP along with methotrexate. Prednisone added for symptom relief.
Output:
{{
  "drugs": ["methotrexate", "prednisone"],
  "regimens": ["r-chop"]
}}
Input: Continued on ABVD and BEACOPP.
Output:
{{
  "drugs": [],
  "regimens": ["abvd", "beacopp"]
}}
Input: Administered capecitabine, bevacizumab, and oxaliplatin.
Output:
{{
  "drugs": ["capecitabine", "bevacizumab", "oxaliplatin"],
  "regimens": []
}}
Input: No active treatment initiated yet.
Output:
{{
  "drugs": [],
  "regimens": []
}}
Process this text:
{text}
JSON Output:""" 