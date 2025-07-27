"""
Data utilities for Project Argus bias detection system.

This module provides utility functions for data processing, text preprocessing,
bias metrics calculation, and data validation.
"""

import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import spacy
from textblob import TextBlob


logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities for bias detection."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Profanity and offensive language patterns (basic list)
        self.profanity_patterns = [
            r'\b(damn|hell|crap)\b',  # Mild profanity
            # Add more patterns as needed for production use
        ]
        
        # PII patterns for anonymization
        self.pii_patterns = {
            'email': self.email_pattern,
            'phone': self.phone_pattern,
            'ssn': self.ssn_pattern,
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub('[URL]', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonymize PII in text.
        
        Args:
            text: Input text to anonymize
        
        Returns:
            Anonymized text
        """
        anonymized = text
        
        for pii_type, pattern in self.pii_patterns.items():
            anonymized = pattern.sub(f'[{pii_type.upper()}]', anonymized)
        
        return anonymized
    
    def extract_entities(self, text: str, nlp_model=None) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            nlp_model: spaCy model for entity extraction
        
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities
            'MONEY': [],
            'DATE': []
        }
        
        if nlp_model:
            try:
                doc = nlp_model(text)
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
            except Exception as e:
                logger.warning(f"Entity extraction failed: {str(e)}")
        
        return entities
    
    def detect_profanity(self, text: str) -> List[str]:
        """
        Detect profanity in text.
        
        Args:
            text: Input text
        
        Returns:
            List of detected profane words
        """
        detected = []
        text_lower = text.lower()
        
        for pattern in self.profanity_patterns:
            matches = re.findall(pattern, text_lower)
            detected.extend(matches)
        
        return detected
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary of readability metrics
        """
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return {
                'flesch_score': 0.0,
                'flesch_grade': 0.0,
                'avg_sentence_length': 0.0,
                'avg_syllables_per_word': 0.0
            }
        
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),
            'flesch_grade': max(0, flesch_grade),
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word
        }
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (approximation)."""
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = 0
        
        for word in words:
            syllables = self._syllables_in_word(word)
            syllable_count += syllables
        
        return syllable_count
    
    def _syllables_in_word(self, word: str) -> int:
        """Count syllables in a single word (approximation)."""
        vowels = "aeiouy"
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)  # Every word has at least one syllable
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using simple TF-IDF approach.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
        
        Returns:
            List of (keyword, score) tuples
        """
        # Simple keyword extraction using word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Filter out common stop words (basic list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Filter and score keywords
        keywords = []
        total_words = len(words)
        
        for word, freq in word_freq.items():
            if word not in stop_words and len(word) > 2:
                # Simple TF score
                tf_score = freq / total_words
                keywords.append((word, tf_score))
        
        # Sort by score and return top-k
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords[:top_k]


class BiasMetrics:
    """Statistical metrics for bias measurement."""
    
    def __init__(self):
        """Initialize bias metrics calculator."""
        pass
    
    def demographic_parity_difference(
        self, 
        y_pred: np.ndarray, 
        sensitive_features: np.ndarray
    ) -> float:
        """
        Calculate demographic parity difference.
        
        Args:
            y_pred: Predicted labels
            sensitive_features: Sensitive attribute values
        
        Returns:
            Demographic parity difference
        """
        groups = np.unique(sensitive_features)
        if len(groups) < 2:
            return 0.0
        
        group_rates = []
        for group in groups:
            group_mask = sensitive_features == group
            if np.sum(group_mask) > 0:
                positive_rate = np.mean(y_pred[group_mask])
                group_rates.append(positive_rate)
        
        if len(group_rates) < 2:
            return 0.0
        
        return max(group_rates) - min(group_rates)
    
    def equalized_odds_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate equalized odds difference (TPR and FPR differences).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive attribute values
        
        Returns:
            Tuple of (TPR difference, FPR difference)
        """
        groups = np.unique(sensitive_features)
        if len(groups) < 2:
            return 0.0, 0.0
        
        tpr_rates = []
        fpr_rates = []
        
        for group in groups:
            group_mask = sensitive_features == group
            
            if np.sum(group_mask) == 0:
                continue
            
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Calculate TPR and FPR
            if len(np.unique(group_y_true)) == 2:
                tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_rates.append(tpr)
                fpr_rates.append(fpr)
        
        tpr_diff = max(tpr_rates) - min(tpr_rates) if len(tpr_rates) >= 2 else 0.0
        fpr_diff = max(fpr_rates) - min(fpr_rates) if len(fpr_rates) >= 2 else 0.0
        
        return tpr_diff, fpr_diff
    
    def disparate_impact_ratio(
        self,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray
    ) -> float:
        """
        Calculate disparate impact ratio.
        
        Args:
            y_pred: Predicted labels
            sensitive_features: Sensitive attribute values
        
        Returns:
            Disparate impact ratio
        """
        groups = np.unique(sensitive_features)
        if len(groups) < 2:
            return 1.0
        
        group_rates = []
        for group in groups:
            group_mask = sensitive_features == group
            if np.sum(group_mask) > 0:
                positive_rate = np.mean(y_pred[group_mask])
                group_rates.append(positive_rate)
        
        if len(group_rates) < 2 or max(group_rates) == 0:
            return 1.0
        
        return min(group_rates) / max(group_rates)
    
    def individual_fairness_score(
        self,
        y_pred: np.ndarray,
        similarity_matrix: np.ndarray,
        threshold: float = 0.9
    ) -> float:
        """
        Calculate individual fairness score based on similar individuals.
        
        Args:
            y_pred: Predicted labels
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for considering individuals similar
        
        Returns:
            Individual fairness score
        """
        n = len(y_pred)
        fairness_violations = 0
        total_comparisons = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    total_comparisons += 1
                    if abs(y_pred[i] - y_pred[j]) > 0.1:  # Threshold for prediction difference
                        fairness_violations += 1
        
        if total_comparisons == 0:
            return 1.0
        
        return 1.0 - (fairness_violations / total_comparisons)
    
    def statistical_significance_test(
        self,
        group1_scores: List[float],
        group2_scores: List[float],
        alpha: float = 0.05
    ) -> Tuple[float, bool]:
        """
        Test statistical significance of bias between two groups.
        
        Args:
            group1_scores: Scores for group 1
            group2_scores: Scores for group 2
            alpha: Significance level
        
        Returns:
            Tuple of (p-value, is_significant)
        """
        if len(group1_scores) < 2 or len(group2_scores) < 2:
            return 1.0, False
        
        try:
            # Use Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                group1_scores, 
                group2_scores, 
                alternative='two-sided'
            )
            
            is_significant = p_value < alpha
            return p_value, is_significant
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {str(e)}")
            return 1.0, False
    
    def calculate_bias_amplification(
        self,
        input_bias_rate: float,
        output_bias_rate: float
    ) -> float:
        """
        Calculate bias amplification factor.
        
        Args:
            input_bias_rate: Bias rate in input data
            output_bias_rate: Bias rate in model output
        
        Returns:
            Bias amplification factor
        """
        if input_bias_rate == 0:
            return 1.0 if output_bias_rate == 0 else float('inf')
        
        return output_bias_rate / input_bias_rate
    
    def intersectional_bias_score(
        self,
        scores: Dict[Tuple[str, ...], List[float]]
    ) -> Dict[str, float]:
        """
        Calculate intersectional bias scores.
        
        Args:
            scores: Dictionary mapping intersectional groups to score lists
        
        Returns:
            Dictionary of intersectional bias metrics
        """
        # Calculate variance across intersectional groups
        all_group_means = []
        group_sizes = []
        
        for group, group_scores in scores.items():
            if len(group_scores) > 0:
                all_group_means.append(np.mean(group_scores))
                group_sizes.append(len(group_scores))
        
        if len(all_group_means) < 2:
            return {'intersectional_variance': 0.0, 'max_disparity': 0.0}
        
        # Calculate metrics
        intersectional_variance = np.var(all_group_means)
        max_disparity = max(all_group_means) - min(all_group_means)
        
        # Weighted average considering group sizes
        weighted_mean = np.average(all_group_means, weights=group_sizes)
        
        return {
            'intersectional_variance': intersectional_variance,
            'max_disparity': max_disparity,
            'weighted_mean': weighted_mean,
            'num_groups': len(all_group_means)
        }


class DataValidator:
    """Data validation utilities for bias detection."""
    
    def __init__(self):
        """Initialize data validator."""
        self.min_sample_size = 10
        self.max_text_length = 10000
        self.supported_languages = ['en', 'es', 'fr', 'de', 'zh', 'ar']
    
    def validate_text_data(self, texts: List[str]) -> Dict[str, Any]:
        """
        Validate text data for bias detection.
        
        Args:
            texts: List of texts to validate
        
        Returns:
            Validation results dictionary
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check if data is provided
        if not texts:
            results['is_valid'] = False
            results['errors'].append("No text data provided")
            return results
        
        # Filter out empty/invalid texts
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        
        if len(valid_texts) == 0:
            results['is_valid'] = False
            results['errors'].append("No valid text data found")
            return results
        
        # Check sample size
        if len(valid_texts) < self.min_sample_size:
            results['warnings'].append(
                f"Sample size ({len(valid_texts)}) is below recommended minimum ({self.min_sample_size})"
            )
        
        # Check text lengths
        text_lengths = [len(text) for text in valid_texts]
        
        if any(length > self.max_text_length for length in text_lengths):
            results['warnings'].append(
                f"Some texts exceed maximum length ({self.max_text_length})"
            )
        
        # Check for duplicates
        unique_texts = set(valid_texts)
        duplicate_ratio = 1 - (len(unique_texts) / len(valid_texts))
        
        if duplicate_ratio > 0.1:  # More than 10% duplicates
            results['warnings'].append(
                f"High duplicate ratio detected: {duplicate_ratio:.1%}"
            )
        
        # Calculate statistics
        results['statistics'] = {
            'total_texts': len(texts),
            'valid_texts': len(valid_texts),
            'avg_length': np.mean(text_lengths),
            'min_length': min(text_lengths),
            'max_length': max(text_lengths),
            'duplicate_ratio': duplicate_ratio,
            'character_distribution': self._analyze_character_distribution(valid_texts)
        }
        
        return results
    
    def validate_bias_results(self, results: 'BiasDetectionResults') -> Dict[str, Any]:
        """
        Validate bias detection results.
        
        Args:
            results: BiasDetectionResults object to validate
        
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        # Check basic completeness
        if results.total_samples_analyzed == 0:
            validation['is_valid'] = False
            validation['errors'].append("No samples were analyzed")
            return validation
        
        # Check consistency
        if results.biased_samples_count > results.total_samples_analyzed:
            validation['is_valid'] = False
            validation['errors'].append("Biased samples count exceeds total samples")
        
        # Check bias rate calculation
        expected_bias_rate = results.biased_samples_count / results.total_samples_analyzed
        if abs(results.bias_rate - expected_bias_rate) > 0.001:
            validation['warnings'].append("Bias rate calculation inconsistency detected")
        
        # Check detection duration
        if results.detection_duration <= 0:
            validation['warnings'].append("Invalid detection duration")
        
        # Calculate quality score
        quality_factors = []
        
        # Sample size factor
        if results.total_samples_analyzed >= 1000:
            quality_factors.append(1.0)
        elif results.total_samples_analyzed >= 100:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)
        
        # Bias instance diversity
        if results.bias_instances:
            bias_types = set(inst.bias_type for inst in results.bias_instances)
            diversity_score = len(bias_types) / len(BiasType)
            quality_factors.append(diversity_score)
        else:
            quality_factors.append(0.0)
        
        # Statistical significance
        if hasattr(results, 'demographic_analysis') and results.demographic_analysis:
            if results.demographic_analysis.statistical_significance:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.7)
        else:
            quality_factors.append(0.5)
        
        validation['quality_score'] = np.mean(quality_factors)
        
        return validation
    
    def _analyze_character_distribution(self, texts: List[str]) -> Dict[str, float]:
        """Analyze character distribution in texts."""
        all_chars = ''.join(texts)
        total_chars = len(all_chars)
        
        if total_chars == 0:
            return {}
        
        char_counts = Counter(all_chars)
        
        return {
            'ascii_ratio': sum(1 for c in all_chars if ord(c) < 128) / total_chars,
            'digit_ratio': sum(1 for c in all_chars if c.isdigit()) / total_chars,
            'punct_ratio': sum(1 for c in all_chars if not c.isalnum() and not c.isspace()) / total_chars,
            'space_ratio': sum(1 for c in all_chars if c.isspace()) / total_chars
        }
    
    def generate_data_quality_report(self, texts: List[str]) -> str:
        """
        Generate a comprehensive data quality report.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            Data quality report as string
        """
        validation = self.validate_text_data(texts)
        preprocessor = TextPreprocessor()
        
        report = []
        report.append("=== PROJECT ARGUS DATA QUALITY REPORT ===\n")
        
        # Overview
        report.append("OVERVIEW:")
        report.append(f"  Total texts: {validation['statistics']['total_texts']}")
        report.append(f"  Valid texts: {validation['statistics']['valid_texts']}")
        report.append(f"  Validation status: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        report.append("")
        
        # Statistics
        stats = validation['statistics']
        report.append("TEXT STATISTICS:")
        report.append(f"  Average length: {stats['avg_length']:.1f} characters")
        report.append(f"  Length range: {stats['min_length']} - {stats['max_length']}")
        report.append(f"  Duplicate ratio: {stats['duplicate_ratio']:.1%}")
        report.append("")
        
        # Character distribution
        char_dist = stats['character_distribution']
        report.append("CHARACTER DISTRIBUTION:")
        report.append(f"  ASCII characters: {char_dist.get('ascii_ratio', 0):.1%}")
        report.append(f"  Digits: {char_dist.get('digit_ratio', 0):.1%}")
        report.append(f"  Punctuation: {char_dist.get('punct_ratio', 0):.1%}")
        report.append("")
        
        # Issues
        if validation['errors']:
            report.append("ERRORS:")
            for error in validation['errors']:
                report.append(f"  - {error}")
            report.append("")
        
        if validation['warnings']:
            report.append("WARNINGS:")
            for warning in validation['warnings']:
                report.append(f"  - {warning}")
            report.append("")
        
        # Sample analysis
        if validation['statistics']['valid_texts'] > 0:
            sample_texts = [text for text in texts if isinstance(text, str) and text.strip()][:5]
            report.append("SAMPLE ANALYSIS:")
            
            for i, text in enumerate(sample_texts, 1):
                readability = preprocessor.calculate_readability(text)
                keywords = preprocessor.extract_keywords(text, top_k=3)
                
                report.append(f"  Sample {i}:")
                report.append(f"    Length: {len(text)} characters")
                report.append(f"    Readability score: {readability['flesch_score']:.1f}")
                report.append(f"    Top keywords: {', '.join([kw[0] for kw in keywords])}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        
        if stats['duplicate_ratio'] > 0.1:
            report.append("  - Consider removing duplicate texts to improve analysis quality")
        
        if validation['statistics']['valid_texts'] < self.min_sample_size:
            report.append(f"  - Increase sample size to at least {self.min_sample_size} for reliable results")
        
        if any(len(text) > self.max_text_length for text in texts if isinstance(text, str)):
            report.append("  - Consider truncating very long texts for better processing performance")
        
        report.append("")
        report.append("=== END REPORT ===")
        
        return '\n'.join(report)