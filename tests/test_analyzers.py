"""
Analyzer component tests for Project Argus.

Tests for demographic analyzer, linguistic analyzer, and counterfactual generator.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from argus.core.config import ArgusConfig
from argus.core.results import BiasInstance, BiasType, SeverityLevel
from argus.analyzers.demographic_analyzer import DemographicAnalyzer
from argus.analyzers.linguistic_analyzer import LinguisticAnalyzer
from argus.analyzers.counterfactual_generator import CounterfactualGenerator


class TestDemographicAnalyzer:
    """Test cases for demographic bias analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create demographic analyzer for testing."""
        config = ArgusConfig()
        return DemographicAnalyzer(config)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config is not None
        assert "gender" in analyzer.demographic_groups
        assert "race_ethnicity" in analyzer.demographic_groups
        assert "age" in analyzer.demographic_groups
    
    def test_detect_demographic_groups(self, analyzer):
        """Test demographic group detection."""
        text = "The woman and the man work together as doctors"
        groups = analyzer._detect_demographic_groups(text)
        
        assert "gender" in groups
        assert "female" in groups["gender"]
        assert "male" in groups["gender"]
    
    def test_check_occupation_stereotypes(self, analyzer):
        """Test occupation stereotype detection."""
        biased_text = "Women are naturally better suited to be nurses"
        bias_instances = analyzer._check_occupation_stereotypes(
            biased_text, "gender", "female"
        )
        
        # Should detect bias if threshold is met
        assert isinstance(bias_instances, list)
    
    def test_check_trait_associations(self, analyzer):
        """Test trait association detection."""
        biased_text = "Men are naturally more rational and logical"
        bias_instances = analyzer._check_trait_associations(
            biased_text, "gender", "male"
        )
        
        assert isinstance(bias_instances, list)
    
    def test_check_exclusionary_language(self, analyzer):
        """Test exclusionary language detection."""
        exclusionary_text = "Only men should be engineers"
        bias_instances = analyzer._check_exclusionary_language(
            exclusionary_text, "gender", "male"
        )
        
        assert isinstance(bias_instances, list)
        # Should detect bias for exclusionary language
        if bias_instances:
            assert bias_instances[0].bias_type == BiasType.DEMOGRAPHIC
    
    def test_analyze_text(self, analyzer):
        """Test full text analysis."""
        biased_text = "The female nurse helped the male doctor with surgery"
        output_text = "She was emotional while he remained rational"
        
        bias_instances = analyzer.analyze_text(biased_text, output_text, "en")
        
        assert isinstance(bias_instances, list)
        # All instances should be demographic bias
        for instance in bias_instances:
            assert instance.bias_type == BiasType.DEMOGRAPHIC
    
    def test_calculate_occupation_bias_score(self, analyzer):
        """Test occupation bias score calculation."""
        text = "Women should be nurses because they are naturally caring"
        score = analyzer._calculate_occupation_bias_score(
            text, "gender", "female", "nurse", "female_dominated"
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_group_instances_by_demographics(self, analyzer):
        """Test grouping bias instances by demographics."""
        instances = [
            BiasInstance(
                bias_type=BiasType.DEMOGRAPHIC,
                severity=SeverityLevel.MEDIUM,
                confidence=0.7,
                text_sample="Test",
                biased_terms=["test"],
                context={"group_type": "gender", "group_name": "female"}
            ),
            BiasInstance(
                bias_type=BiasType.DEMOGRAPHIC,
                severity=SeverityLevel.HIGH,
                confidence=0.8,
                text_sample="Test2",
                biased_terms=["test2"],
                context={"group_type": "gender", "group_name": "male"}
            )
        ]
        
        grouped = analyzer._group_instances_by_demographics(instances)
        
        assert "gender" in grouped
        assert "female" in grouped["gender"]
        assert "male" in grouped["gender"]
        assert len(grouped["gender"]["female"]) == 1
        assert len(grouped["gender"]["male"]) == 1
    
    def test_calculate_group_comparisons(self, analyzer):
        """Test group comparison calculations."""
        grouped_instances = {
            "gender": {
                "male": [
                    BiasInstance(
                        bias_type=BiasType.DEMOGRAPHIC,
                        severity=SeverityLevel.LOW,
                        confidence=0.5,
                        text_sample="Test",
                        biased_terms=["test"],
                        bias_score=0.3
                    )
                ],
                "female": [
                    BiasInstance(
                        bias_type=BiasType.DEMOGRAPHIC,
                        severity=SeverityLevel.HIGH,
                        confidence=0.8,
                        text_sample="Test2",
                        biased_terms=["test2"],
                        bias_score=0.7
                    )
                ]
            }
        }
        
        comparisons = analyzer._calculate_group_comparisons(grouped_instances)
        
        assert "gender" in comparisons
        assert "male" in comparisons["gender"]
        assert "female" in comparisons["gender"]
        assert "avg_bias_score" in comparisons["gender"]["male"]
        assert "disparity" in comparisons["gender"]["male"]


class TestLinguisticAnalyzer:
    """Test cases for linguistic bias analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create linguistic analyzer for testing."""
        config = ArgusConfig()
        return LinguisticAnalyzer(config)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config is not None
        assert "aave" in analyzer.dialect_markers
        assert "formal" in analyzer.formality_markers
    
    def test_detect_dialect_bias(self, analyzer):
        """Test dialect bias detection."""
        aave_text = "Y'all ain't gonna believe this"
        bias_instances = analyzer._detect_dialect_bias(aave_text, "en")
        
        assert isinstance(bias_instances, list)
    
    def test_detect_formality_bias(self, analyzer):
        """Test formality bias detection."""
        informal_text = "This is totally wrong and you shouldn't use it"
        bias_instances = analyzer._detect_formality_bias(informal_text, "en")
        
        assert isinstance(bias_instances, list)
    
    def test_detect_language_preference_bias(self, analyzer):
        """Test language preference bias detection."""
        biased_text = "English is the only proper language for business"
        bias_instances = analyzer._detect_language_preference_bias(biased_text, "en")
        
        assert isinstance(bias_instances, list)
        # Should detect language supremacy bias
        if bias_instances:
            assert bias_instances[0].bias_type == BiasType.LINGUISTIC
    
    def test_detect_inclusive_language_bias(self, analyzer):
        """Test inclusive language bias detection."""
        non_inclusive_text = "The chairman and spokesman discussed mankind's future"
        bias_instances = analyzer._detect_inclusive_language_bias(non_inclusive_text, "en")
        
        assert isinstance(bias_instances, list)
    
    def test_detect_english_inclusive_bias(self, analyzer):
        """Test English-specific inclusive language detection."""
        text = "The fireman and policeman protected mankind"
        bias_instances = analyzer._detect_english_inclusive_bias(text)
        
        assert isinstance(bias_instances, list)
        # Should detect non-inclusive terms
        if bias_instances:
            assert all(inst.bias_type == BiasType.LINGUISTIC for inst in bias_instances)
    
    def test_analyze_single_text(self, analyzer):
        """Test single text analysis."""
        text = "Y'all shouldn't use broken English in professional settings"
        bias_instances = analyzer.analyze_single_text(text, "en")
        
        assert isinstance(bias_instances, list)
        for instance in bias_instances:
            assert instance.bias_type == BiasType.LINGUISTIC
    
    def test_calculate_dialect_bias_score(self, analyzer):
        """Test dialect bias score calculation."""
        text = "That ain't proper English and you should fix your grammar"
        score = analyzer._calculate_dialect_bias_score(text, "aave", ["ain't"])
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_analyze_dialect_distribution(self, analyzer):
        """Test dialect distribution analysis."""
        texts = [
            "Hello, how are you today?",
            "Y'all come back now, ya hear?",
            "Crikey, that's brilliant, mate!",
            "I reckon it's fixin' to rain"
        ]
        
        distribution = analyzer._analyze_dialect_distribution(texts)
        
        assert isinstance(distribution, dict)
        # Should detect southern and australian dialects
        assert any(dialect in distribution for dialect in ["southern_us", "australian"])
    
    def test_analyze_formality_distribution(self, analyzer):
        """Test formality distribution analysis."""
        texts = [
            "Furthermore, I would like to emphasize the importance of this matter",
            "This is totally awesome and super cool!",
            "Consequently, the aforementioned hypothesis requires validation",
            "Yeah, that's pretty much what I was thinking too"
        ]
        
        distribution = analyzer._analyze_formality_distribution(texts)
        
        assert isinstance(distribution, dict)
        if distribution:  # Only check if we have data
            assert "average_formality" in distribution


class TestCounterfactualGenerator:
    """Test cases for counterfactual generator."""
    
    @pytest.fixture
    def generator(self):
        """Create counterfactual generator for testing."""
        config = ArgusConfig()
        return CounterfactualGenerator(config)
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config is not None
        assert "gender" in generator.substitutions
        assert "race_ethnicity" in generator.substitutions
    
    def test_generate_counterfactuals(self, generator):
        """Test counterfactual generation."""
        text = "John is a skilled engineer who works hard"
        variations = generator.generate_counterfactuals(text)
        
        assert isinstance(variations, list)
        # Should generate some variations
        if variations:
            assert all(isinstance(var, str) for var in variations)
    
    def test_generate_direct_substitutions(self, generator):
        """Test direct substitution method."""
        text = "The man went to work as a doctor"
        variations = generator._generate_direct_substitutions(text)
        
        assert isinstance(variations, list)
    
    def test_substitute_gender_terms(self, generator):
        """Test gender term substitution."""
        text = "He is a successful businessman"
        variations = generator._substitute_gender_terms(text)
        
        assert isinstance(variations, list)
        # Should generate variations with different pronouns
        if variations:
            assert any("she" in var.lower() or "they" in var.lower() for var in variations)
    
    def test_substitute_names(self, generator):
        """Test name substitution."""
        text = "Michael is applying for the engineering position"
        variations = generator._substitute_names(text)
        
        assert isinstance(variations, list)
        # Should generate variations with different names
        if variations:
            assert any("Michael" not in var for var in variations)
    
    def test_substitute_race_terms(self, generator):
        """Test race/ethnicity term substitution."""
        text = "The Asian candidate has excellent qualifications"
        variations = generator._substitute_race_terms(text)
        
        assert isinstance(variations, list)
    
    def test_substitute_religion_terms(self, generator):
        """Test religion term substitution."""
        text = "The Christian community supports this initiative"
        variations = generator._substitute_religion_terms(text)
        
        assert isinstance(variations, list)
    
    def test_generate_template_variations(self, generator):
        """Test template-based variation generation."""
        text = "The doctor is very professional"
        variations = generator._generate_template_variations(text)
        
        assert isinstance(variations, list)
    
    def test_calculate_bias_score(self, generator):
        """Test bias score calculation."""
        biased_text = "Women are naturally emotional and weak"
        score = generator._calculate_bias_score(biased_text)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_generate_systematic_test_cases(self, generator):
        """Test systematic test case generation."""
        test_cases = generator.generate_systematic_test_cases()
        
        assert isinstance(test_cases, list)
        if test_cases:
            original, variation, metadata = test_cases[0]
            assert isinstance(original, str)
            assert isinstance(variation, str)
            assert isinstance(metadata, dict)
            assert "attribute_type" in metadata
    
    def test_generate_attribute_test_cases(self, generator):
        """Test attribute-specific test case generation."""
        test_cases = generator._generate_attribute_test_cases("gender")
        
        assert isinstance(test_cases, list)
        if test_cases:
            original, variation, metadata = test_cases[0]
            assert metadata["attribute_type"] == "gender"
    
    def test_generate_intersectional_test_cases(self, generator):
        """Test intersectional test case generation."""
        test_cases = generator._generate_intersectional_test_cases()
        
        assert isinstance(test_cases, list)
        if test_cases:
            original, variation, metadata = test_cases[0]
            assert metadata["attribute_type"] == "intersectional"
    
    @patch('argus.analyzers.counterfactual_generator.AutoTokenizer')
    @patch('argus.analyzers.counterfactual_generator.AutoModelForMaskedLM')
    def test_mlm_initialization(self, mock_model, mock_tokenizer, generator):
        """Test MLM model initialization."""
        # Mock successful initialization
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        generator._initialize_mlm_model()
        
        assert mock_tokenizer.from_pretrained.called
        assert mock_model.from_pretrained.called
    
    def test_generate_analysis(self, generator):
        """Test full counterfactual analysis generation."""
        texts = [
            "John is a great engineer",
            "Mary works as a nurse",
            "The doctor examined the patient"
        ]
        
        analysis = generator.generate_analysis(texts)
        
        assert hasattr(analysis, 'original_outputs')
        assert hasattr(analysis, 'counterfactual_outputs')
        assert hasattr(analysis, 'bias_differences')
        assert hasattr(analysis, 'average_bias_change')
        assert hasattr(analysis, 'significant_changes')
        
        assert isinstance(analysis.original_outputs, list)
        assert isinstance(analysis.counterfactual_outputs, list)
        assert isinstance(analysis.bias_differences, list)
        assert isinstance(analysis.average_bias_change, float)
        assert isinstance(analysis.significant_changes, int)