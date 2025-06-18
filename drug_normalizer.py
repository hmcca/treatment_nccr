import pandas as pd
from fuzzywuzzy import fuzz
from typing import List, Dict, Set
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugNormalizer:
    def __init__(self, synonym_file: str, threshold: float = 85):
        """
        Initialize the drug normalizer with a synonym file and matching threshold.
        
        Args:
            synonym_file: Path to the CSV file containing drug synonyms
            threshold: Fuzzy matching threshold (0-100)
        """
        self.threshold = threshold
        self.synonym_map = self._load_synonyms(synonym_file)
        logger.info(f"Loaded {len(self.synonym_map)} drug synonyms")
        
    def _load_synonyms(self, synonym_file: str) -> Dict[str, str]:
        """Load and process the drug synonym file."""
        df = pd.read_csv(synonym_file)
        synonym_map = {}
        
        for _, row in df.iterrows():
            generic_name = row['DrugName'].strip()
            synonyms = [s.strip() for s in row['DrugSynonym'].split(',')]
            
            # Add the generic name itself as a synonym
            synonyms.append(generic_name)
            
            # Add all synonyms to the map (all in lowercase)
            for synonym in synonyms:
                if synonym:  # Skip empty strings
                    synonym_map[synonym.lower()] = generic_name
        
        return synonym_map
    
    def _find_best_match(self, drug_name: str) -> str:
        """
        Find the best matching generic name for a given drug name.
        
        Args:
            drug_name: The drug name to normalize
            
        Returns:
            The best matching generic name, or the original name if no good match is found
        """
        if not isinstance(drug_name, str) or not drug_name.strip():
            return drug_name
            
        drug_name = drug_name.strip().lower()
        
        # First try exact match
        if drug_name in self.synonym_map:
            return self.synonym_map[drug_name]
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        for synonym, generic in self.synonym_map.items():
            score = fuzz.ratio(drug_name, synonym)
            if score > best_score:
                best_score = score
                best_match = generic
        
        if best_score >= self.threshold:
            logger.debug(f"Matched '{drug_name}' to '{best_match}' with score {best_score}")
            return best_match
        else:
            logger.debug(f"No good match found for '{drug_name}' (best score: {best_score})")
            return drug_name
    
    def normalize_drugs(self, drugs: List[str]) -> List[str]:
        """
        Normalize a list of drug names to their generic equivalents.
        
        Args:
            drugs: List of drug names to normalize
            
        Returns:
            List of normalized drug names
        """
        if not drugs:
            return []
            
        normalized = []
        for drug in drugs:
            if isinstance(drug, str):
                normalized_drug = self._find_best_match(drug)
                normalized.append(normalized_drug)
            else:
                normalized.append(drug)
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in normalized if not (x in seen or seen.add(x))]
    
    def normalize_drugs_json(self, drugs_json: str) -> str:
        """
        Normalize drugs from a JSON string.
        
        Args:
            drugs_json: JSON string containing drug names
            
        Returns:
            JSON string with normalized drug names
        """
        try:
            drugs = json.loads(drugs_json)
            normalized = self.normalize_drugs(drugs)
            return json.dumps(normalized)
        except json.JSONDecodeError:
            return drugs_json

class RegimenDrugMapper:
    def __init__(self, regimen_drug_file: str, threshold: int = 70):
        """
        Initialize the regimen drug mapper with a mapping file.
        
        Args:
            regimen_drug_file: Path to the CSV file containing regimen to drug mappings
            threshold: Fuzzy matching threshold (0-100)
        """
        self.threshold = threshold
        self.regimen_map = self._load_regimen_map(regimen_drug_file)
        logger.info(f"Loaded {len(self.regimen_map)} regimen mappings")
        
    def _load_regimen_map(self, regimen_drug_file: str) -> Dict[str, List[str]]:
        """Load regimen to drug mapping from file."""
        df = pd.read_csv(regimen_drug_file)
        regimen_map = {}
        for _, row in df.iterrows():
            regimen = str(row['Regimen']).strip().lower()
            drugs = [d.strip().lower() for d in str(row['Drugs']).split(',')]
            regimen_map[regimen] = drugs
        return regimen_map
    
    def _normalize_regimen_name(self, regimen: str) -> str:
        """Normalize a regimen name for better matching."""
        if not isinstance(regimen, str):
            return ""
        # Convert to lowercase and strip
        regimen = regimen.strip().lower()
        # Remove common variations
        regimen = regimen.replace("regimen", "").strip()
        regimen = regimen.replace("therapy", "").strip()
        regimen = regimen.replace("protocol", "").strip()
        return regimen
    
    def _fuzzy_match_regimen(self, regimen: str) -> str:
        """
        Find the best matching regimen using fuzzy matching.
        
        Args:
            regimen: The regimen name to match
            
        Returns:
            The best matching regimen name from the mapping, or None if no good match
        """
        if not regimen:
            return None
            
        normalized_regimen = self._normalize_regimen_name(regimen)
        best_match = None
        best_score = 0
        
        # First try exact match
        if normalized_regimen in self.regimen_map:
            logger.debug(f"Exact match found for regimen: {regimen}")
            return normalized_regimen
        
        # Try fuzzy matching
        for known_regimen in self.regimen_map.keys():
            # Try different fuzzy matching methods
            ratio_score = fuzz.ratio(normalized_regimen, known_regimen)
            partial_score = fuzz.partial_ratio(normalized_regimen, known_regimen)
            token_sort_score = fuzz.token_sort_ratio(normalized_regimen, known_regimen)
            
            # Use the best score among the different methods
            score = max(ratio_score, partial_score, token_sort_score)
            
            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = known_regimen
        
        if best_match:
            logger.debug(f"Matched '{regimen}' to '{best_match}' with score {best_score}")
        else:
            logger.debug(f"No good match found for '{regimen}' (best score: {best_score})")
        
        return best_match
    
    def get_drugs_from_regimen(self, regimen: str) -> List[str]:
        """
        Get constituent drugs for a given regimen using fuzzy matching.
        
        Args:
            regimen: The regimen name to look up
            
        Returns:
            List of constituent drugs for the regimen
        """
        if not regimen:
            return []
            
        matched_regimen = self._fuzzy_match_regimen(regimen)
        if matched_regimen:
            drugs = self.regimen_map[matched_regimen]
            logger.debug(f"Found {len(drugs)} drugs for regimen '{regimen}' -> '{matched_regimen}'")
            return drugs
        else:
            logger.debug(f"No drugs found for regimen: {regimen}")
            return []
    
    def get_combined_drugs(self, extracted_drugs: List[str], extracted_regimens: List[str]) -> List[str]:
        """
        Get combined unique drugs from both extracted drugs and regimens.
        
        Args:
            extracted_drugs: List of directly extracted drugs
            extracted_regimens: List of extracted regimens
            
        Returns:
            List of unique drugs from both sources
        """
        # Get drugs from regimens
        regimen_drugs = []
        for regimen in extracted_regimens:
            if regimen:
                drugs = self.get_drugs_from_regimen(regimen)
                regimen_drugs.extend(drugs)
                logger.debug(f"Regimen '{regimen}' mapped to drugs: {drugs}")
        
        # Combine with extracted drugs and get unique set
        all_drugs = set()
        for drug in extracted_drugs + regimen_drugs:
            if drug:
                all_drugs.add(str(drug).strip().lower())
        
        logger.debug(f"Combined {len(extracted_drugs)} extracted drugs with {len(regimen_drugs)} regimen drugs")
        logger.debug(f"Total unique drugs: {len(all_drugs)}")
        
        return list(all_drugs) 