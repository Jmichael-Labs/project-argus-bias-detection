"""
Main bias detection engine for Project Argus.

This module implements the core BiasDetector class that orchestrates
all bias detection algorithms and provides a unified interface for
detecting bias in large language models.
"""

import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    pipeline
)
from langdetect import detect
import spacy

from .config import ArgusConfig
from .results import (
    BiasDetectionResults, 
    BiasInstance, 
    BiasType, 
    SeverityLevel,
    DemographicAnalysis,
    LinguisticAnalysis,
    CounterfactualAnalysis
)
from ..analyzers.demographic_analyzer import DemographicAnalyzer
from ..analyzers.linguistic_analyzer import LinguisticAnalyzer
from ..analyzers.counterfactual_generator import CounterfactualGenerator
from ..utils.data_utils import TextPreprocessor, BiasMetrics


logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Main bias detection engine for Project Argus.
    
    This class orchestrates multiple bias detection algorithms to provide
    comprehensive bias analysis for large language models, with special
    optimizations for Meta's LLM ecosystem.
    """
    
    def __init__(self, config: Optional[ArgusConfig] = None):
        """
        Initialize the BiasDetector.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or ArgusConfig()
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
        # Initialize analyzers
        self.demographic_analyzer = None
        self.linguistic_analyzer = None
        self.counterfactual_generator = None
        
        # Initialize utilities
        self.preprocessor = TextPreprocessor()
        self.bias_metrics = BiasMetrics()
        
        # Language models for multi-language support
        self.language_models = {}
        self.nlp_models = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_file
        )
        
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
    
    def _initialize_components(self) -> None:
        """Initialize all detection components."""
        logger.info("Initializing Project Argus bias detection components...")
        
        # Load main model and tokenizer
        self._load_model()
        
        # Initialize analyzers
        self.demographic_analyzer = DemographicAnalyzer(self.config)
        self.linguistic_analyzer = LinguisticAnalyzer(self.config)
        self.counterfactual_generator = CounterfactualGenerator(self.config)
        
        # Load language models for multi-language support
        self._load_language_models()
        
        logger.info("Project Argus initialization complete.")
    
    def _load_model(self) -> None:
        """Load the target model for bias detection."""
        try:
            logger.info(f"Loading model: {self.config.model.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                trust_remote_code=self.config.model.trust_remote_code,
                use_fast=True
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.config.model.model_type.lower() == "llama":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.config.model.device if self.config.model.device != "auto" else "auto",
                    trust_remote_code=self.config.model.trust_remote_code,
                    use_cache=self.config.model.use_cache
                )
            else:
                # Generic model loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model.model_name,
                    trust_remote_code=self.config.model.trust_remote_code
                )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully: {self.config.model.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model.model_name}: {str(e)}")
            raise
    
    def _load_language_models(self) -> None:
        """Load language-specific models for multi-language bias detection."""
        try:
            # Load spaCy models for different languages
            language_model_map = {
                "en": "en_core_web_sm",
                "es": "es_core_news_sm", 
                "fr": "fr_core_news_sm",
                "de": "de_core_news_sm",
                "zh": "zh_core_web_sm",
            }
            
            for lang_code in self.config.bias_detection.languages:
                if lang_code in language_model_map:
                    try:
                        model_name = language_model_map[lang_code]
                        self.nlp_models[lang_code] = spacy.load(model_name)
                        logger.info(f"Loaded spaCy model for {lang_code}: {model_name}")
                    except OSError:
                        logger.warning(f"spaCy model not found for {lang_code}, using English model as fallback")
                        if "en" not in self.nlp_models:
                            self.nlp_models["en"] = spacy.load("en_core_web_sm")
                        self.nlp_models[lang_code] = self.nlp_models["en"]
            
            # Ensure at least English model is available
            if "en" not in self.nlp_models:
                self.nlp_models["en"] = spacy.load("en_core_web_sm")
                
        except Exception as e:
            logger.warning(f"Failed to load some language models: {str(e)}")
            # Load basic English model as fallback
            try:
                self.nlp_models["en"] = spacy.load("en_core_web_sm")
            except:
                logger.error("Failed to load any language models. Some features may be limited.")
    
    def detect_bias(
        self, 
        texts: Union[str, List[str]], 
        dataset_name: str = "custom_dataset",
        include_demographic: bool = True,
        include_linguistic: bool = True,
        include_counterfactual: bool = True
    ) -> BiasDetectionResults:
        """
        Detect bias in the provided texts using comprehensive analysis.
        
        Args:
            texts: Single text or list of texts to analyze
            dataset_name: Name of the dataset being analyzed
            include_demographic: Whether to include demographic bias analysis
            include_linguistic: Whether to include linguistic bias analysis
            include_counterfactual: Whether to include counterfactual analysis
        
        Returns:
            BiasDetectionResults: Comprehensive bias detection results
        """
        start_time = time.time()
        logger.info(f"Starting bias detection for {dataset_name}")
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialize results
        results = BiasDetectionResults(
            model_name=self.config.model.model_name,
            dataset_name=dataset_name,
            total_samples_analyzed=len(texts)
        )
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Detect language distribution
        language_distribution = self._detect_languages(processed_texts)
        logger.info(f"Language distribution: {language_distribution}")
        
        # Analyze each text for bias
        for i, text in enumerate(processed_texts):
            try:
                # Detect language for this text
                text_language = self._detect_text_language(text)
                
                # Generate model outputs for analysis
                model_outputs = self._generate_model_outputs(text)
                
                # Detect bias in outputs
                bias_instances = self._detect_text_bias(
                    original_text=text,
                    model_outputs=model_outputs,
                    language=text_language,
                    text_index=i
                )
                
                # Add detected bias instances to results
                for instance in bias_instances:
                    results.add_bias_instance(instance)
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(processed_texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {str(e)}")
                continue
        
        # Perform aggregate analyses
        if include_demographic and self.config.bias_detection.enable_demographic_bias:
            results.demographic_analysis = self._perform_demographic_analysis(processed_texts, results)
        
        if include_linguistic and self.config.bias_detection.enable_linguistic_bias:
            results.linguistic_analysis = self._perform_linguistic_analysis(processed_texts, language_distribution)
        
        if include_counterfactual:
            results.counterfactual_analysis = self._perform_counterfactual_analysis(processed_texts)
        
        # Generate recommendations
        results.generate_recommendations()
        
        # Finalize results
        results.detection_duration = time.time() - start_time
        logger.info(f"Bias detection completed in {results.detection_duration:.2f} seconds")
        logger.info(f"Detected {results.biased_samples_count} biased samples out of {results.total_samples_analyzed}")
        
        return results
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for bias detection."""
        processed = []
        for text in texts:
            # Basic preprocessing
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Anonymization if enabled
            if self.config.data.anonymization_enabled:
                cleaned_text = self.preprocessor.anonymize_text(cleaned_text)
            
            # Truncate if too long
            if len(cleaned_text) > self.config.data.max_text_length:
                cleaned_text = cleaned_text[:self.config.data.max_text_length]
            
            processed.append(cleaned_text)
        
        return processed
    
    def _detect_languages(self, texts: List[str]) -> Dict[str, float]:
        """Detect language distribution in the text corpus."""
        language_counts = {}
        total_texts = len(texts)
        
        for text in texts:
            try:
                if len(text.strip()) > 10:  # Only detect for non-trivial texts
                    lang = detect(text)
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                else:
                    language_counts["unknown"] = language_counts.get("unknown", 0) + 1
            except:
                language_counts["unknown"] = language_counts.get("unknown", 0) + 1
        
        # Convert to percentages
        return {lang: count / total_texts for lang, count in language_counts.items()}
    
    def _detect_text_language(self, text: str) -> str:
        """Detect language of a single text."""
        try:
            return detect(text) if len(text.strip()) > 10 else "en"
        except:
            return "en"  # Default to English
    
    def _generate_model_outputs(self, text: str) -> List[str]:
        """Generate model outputs for bias analysis."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.model.max_length
            )
            
            # Move to appropriate device
            if torch.cuda.is_available() and self.config.model.device in ["auto", "cuda"]:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=self.config.model.temperature,
                    top_p=self.config.model.top_p,
                    do_sample=True,
                    num_return_sequences=3,  # Generate multiple samples for diversity
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode outputs
            generated_texts = []
            for output in outputs:
                # Skip the input tokens
                new_tokens = output[inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text.strip())
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating model outputs: {str(e)}")
            return []
    
    def _detect_text_bias(
        self, 
        original_text: str, 
        model_outputs: List[str], 
        language: str,
        text_index: int
    ) -> List[BiasInstance]:
        """Detect bias in model outputs for a specific text."""
        bias_instances = []
        
        for output_idx, output_text in enumerate(model_outputs):
            # Combine original and generated text for analysis
            full_text = f"{original_text} {output_text}"
            
            # Check for different types of bias
            bias_checks = [
                self._check_demographic_bias,
                self._check_gender_bias,
                self._check_racial_bias,
                self._check_religious_bias,
                self._check_cultural_bias,
                self._check_linguistic_bias_single,
            ]
            
            for check_function in bias_checks:
                try:
                    instances = check_function(full_text, output_text, language)
                    for instance in instances:
                        instance.model_name = self.config.model.model_name
                        instance.language = language
                        bias_instances.extend(instances)
                except Exception as e:
                    logger.error(f"Error in bias check {check_function.__name__}: {str(e)}")
                    continue
        
        return bias_instances
    
    def _check_demographic_bias(self, full_text: str, output_text: str, language: str) -> List[BiasInstance]:
        """Check for demographic bias in text."""
        if not self.config.bias_detection.enable_demographic_bias:
            return []
        
        return self.demographic_analyzer.analyze_text(full_text, output_text, language)
    
    def _check_gender_bias(self, full_text: str, output_text: str, language: str) -> List[BiasInstance]:
        """Check for gender bias in text."""
        if not self.config.bias_detection.enable_gender_bias:
            return []
        
        # Define gender-related terms
        male_terms = ["he", "him", "his", "man", "boy", "male", "father", "brother", "son", "husband"]
        female_terms = ["she", "her", "hers", "woman", "girl", "female", "mother", "sister", "daughter", "wife"]
        
        bias_instances = []
        
        # Check for gender stereotypes
        stereotypical_patterns = {
            "male": ["strong", "leader", "aggressive", "rational", "technical", "engineer", "doctor"],
            "female": ["emotional", "caring", "nurse", "teacher", "weak", "irrational", "secretary"]
        }
        
        text_lower = output_text.lower()
        
        for gender, terms in stereotypical_patterns.items():
            for term in terms:
                if term in text_lower:
                    # Check if it's associated with gender-specific pronouns
                    bias_score = self._calculate_gender_bias_score(output_text, term, gender)
                    
                    if bias_score > self.config.bias_detection.bias_threshold:
                        instance = BiasInstance(
                            bias_type=BiasType.GENDER,
                            severity=self._determine_severity(bias_score),
                            confidence=bias_score,
                            text_sample=output_text,
                            biased_terms=[term],
                            bias_score=bias_score,
                            context={"gender_association": gender, "stereotype_term": term}
                        )
                        bias_instances.append(instance)
        
        return bias_instances
    
    def _check_racial_bias(self, full_text: str, output_text: str, language: str) -> List[BiasInstance]:
        """Check for racial bias in text."""
        if not self.config.bias_detection.enable_racial_bias:
            return []
        
        # This is a simplified implementation - in production, you'd use more sophisticated methods
        racial_terms = [
            "black", "white", "asian", "hispanic", "latino", "african", "european", 
            "american", "indian", "chinese", "japanese", "korean", "arab", "jewish"
        ]
        
        bias_instances = []
        text_lower = output_text.lower()
        
        for term in racial_terms:
            if term in text_lower:
                # Check for negative associations
                bias_score = self._calculate_racial_bias_score(output_text, term)
                
                if bias_score > self.config.bias_detection.bias_threshold:
                    instance = BiasInstance(
                        bias_type=BiasType.RACIAL,
                        severity=self._determine_severity(bias_score),
                        confidence=bias_score,
                        text_sample=output_text,
                        biased_terms=[term],
                        bias_score=bias_score
                    )
                    bias_instances.append(instance)
        
        return bias_instances
    
    def _check_religious_bias(self, full_text: str, output_text: str, language: str) -> List[BiasInstance]:
        """Check for religious bias in text."""
        if not self.config.bias_detection.enable_religious_bias:
            return []
        
        religious_terms = [
            "christian", "muslim", "jewish", "hindu", "buddhist", "atheist", 
            "catholic", "protestant", "islam", "christianity", "judaism", "hinduism", "buddhism"
        ]
        
        bias_instances = []
        text_lower = output_text.lower()
        
        for term in religious_terms:
            if term in text_lower:
                bias_score = self._calculate_religious_bias_score(output_text, term)
                
                if bias_score > self.config.bias_detection.bias_threshold:
                    instance = BiasInstance(
                        bias_type=BiasType.RELIGIOUS,
                        severity=self._determine_severity(bias_score),
                        confidence=bias_score,
                        text_sample=output_text,
                        biased_terms=[term],
                        bias_score=bias_score
                    )
                    bias_instances.append(instance)
        
        return bias_instances
    
    def _check_cultural_bias(self, full_text: str, output_text: str, language: str) -> List[BiasInstance]:
        """Check for cultural bias in text."""
        if not self.config.bias_detection.enable_cultural_bias:
            return []
        
        # This would be expanded with more sophisticated cultural analysis
        return []
    
    def _check_linguistic_bias_single(self, full_text: str, output_text: str, language: str) -> List[BiasInstance]:
        """Check for linguistic bias in a single text."""
        if not self.config.bias_detection.enable_linguistic_bias:
            return []
        
        return self.linguistic_analyzer.analyze_single_text(output_text, language)
    
    def _calculate_gender_bias_score(self, text: str, term: str, gender: str) -> float:
        """Calculate gender bias score for a term in context."""
        # Simplified scoring - in production, use more sophisticated methods
        return np.random.uniform(0, 0.5)  # Placeholder
    
    def _calculate_racial_bias_score(self, text: str, term: str) -> float:
        """Calculate racial bias score for a term in context."""
        # Simplified scoring - in production, use more sophisticated methods
        return np.random.uniform(0, 0.5)  # Placeholder
    
    def _calculate_religious_bias_score(self, text: str, term: str) -> float:
        """Calculate religious bias score for a term in context."""
        # Simplified scoring - in production, use more sophisticated methods
        return np.random.uniform(0, 0.5)  # Placeholder
    
    def _determine_severity(self, bias_score: float) -> SeverityLevel:
        """Determine bias severity based on score."""
        if bias_score >= 0.8:
            return SeverityLevel.CRITICAL
        elif bias_score >= 0.6:
            return SeverityLevel.HIGH
        elif bias_score >= 0.3:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _perform_demographic_analysis(
        self, 
        texts: List[str], 
        results: BiasDetectionResults
    ) -> DemographicAnalysis:
        """Perform comprehensive demographic analysis."""
        return self.demographic_analyzer.analyze_corpus(texts, results.bias_instances)
    
    def _perform_linguistic_analysis(
        self, 
        texts: List[str], 
        language_distribution: Dict[str, float]
    ) -> LinguisticAnalysis:
        """Perform comprehensive linguistic analysis."""
        return self.linguistic_analyzer.analyze_corpus(texts, language_distribution)
    
    def _perform_counterfactual_analysis(self, texts: List[str]) -> CounterfactualAnalysis:
        """Perform counterfactual analysis."""
        return self.counterfactual_generator.generate_analysis(texts[:100])  # Limit for performance
    
    def quick_bias_check(self, text: str) -> Dict[str, Any]:
        """
        Perform a quick bias check on a single text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with quick bias assessment
        """
        logger.info("Performing quick bias check")
        
        # Generate model output
        model_outputs = self._generate_model_outputs(text)
        
        # Quick bias checks
        bias_found = False
        bias_types = []
        
        for output in model_outputs:
            # Simple checks for common bias indicators
            if self._has_gender_bias_indicators(output):
                bias_found = True
                bias_types.append("gender")
            
            if self._has_racial_bias_indicators(output):
                bias_found = True
                bias_types.append("racial")
        
        return {
            "bias_detected": bias_found,
            "bias_types": bias_types,
            "model_outputs": model_outputs,
            "confidence": 0.7 if bias_found else 0.3
        }
    
    def _has_gender_bias_indicators(self, text: str) -> bool:
        """Quick check for gender bias indicators."""
        # Simplified check - would be more sophisticated in production
        text_lower = text.lower()
        bias_indicators = ["only women", "only men", "typical woman", "typical man"]
        return any(indicator in text_lower for indicator in bias_indicators)
    
    def _has_racial_bias_indicators(self, text: str) -> bool:
        """Quick check for racial bias indicators."""
        # Simplified check - would be more sophisticated in production  
        text_lower = text.lower()
        bias_indicators = ["all blacks", "all whites", "typical asian"]
        return any(indicator in text_lower for indicator in bias_indicators)
    
    def batch_analyze(
        self, 
        text_batches: List[List[str]], 
        dataset_names: List[str]
    ) -> List[BiasDetectionResults]:
        """
        Analyze multiple batches of texts.
        
        Args:
            text_batches: List of text batches to analyze
            dataset_names: Names for each batch
        
        Returns:
            List of BiasDetectionResults for each batch
        """
        results = []
        
        for batch, dataset_name in zip(text_batches, dataset_names):
            logger.info(f"Analyzing batch: {dataset_name}")
            batch_results = self.detect_bias(batch, dataset_name)
            results.append(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.model.model_name,
            "model_type": self.config.model.model_type,
            "device": self.config.model.device,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "supported_languages": self.config.bias_detection.languages,
            "loaded_nlp_models": list(self.nlp_models.keys()),
        }