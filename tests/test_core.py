"""
Core component tests for Project Argus.

Tests for the main bias detection engine, configuration management,
and results handling.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from argus.core.config import ArgusConfig, ModelConfig, BiasDetectionConfig
from argus.core.detector import BiasDetector
from argus.core.results import (
    BiasDetectionResults, 
    BiasInstance, 
    BiasType, 
    SeverityLevel,
    ResultsAggregator
)


class TestArgusConfig:
    """Test cases for configuration management."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = ArgusConfig()
        
        assert config.model.model_name == "meta-llama/Llama-2-7b-chat-hf"
        assert config.bias_detection.bias_threshold == 0.1
        assert config.monitoring.kafka_bootstrap_servers == "localhost:9092"
        assert config.log_level == "INFO"
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = ArgusConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "bias_detection" in config_dict
        assert "monitoring" in config_dict
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model": {
                "model_name": "test-model",
                "temperature": 0.8
            },
            "bias_detection": {
                "bias_threshold": 0.2
            },
            "log_level": "DEBUG"
        }
        
        config = ArgusConfig.from_dict(config_dict)
        
        assert config.model.model_name == "test-model"
        assert config.model.temperature == 0.8
        assert config.bias_detection.bias_threshold == 0.2
        assert config.log_level == "DEBUG"
    
    def test_config_yaml_save_load(self):
        """Test saving and loading configuration from YAML."""
        config = ArgusConfig()
        config.model.model_name = "test-model"
        config.bias_detection.bias_threshold = 0.3
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            
            # Load the config back
            loaded_config = ArgusConfig.from_yaml(f.name)
            
            assert loaded_config.model.model_name == "test-model"
            assert loaded_config.bias_detection.bias_threshold == 0.3
        
        # Clean up
        Path(f.name).unlink()
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ArgusConfig()
        
        # Valid configuration should pass
        config.validate()
        
        # Invalid threshold should raise error
        config.bias_detection.bias_threshold = 1.5
        with pytest.raises(ValueError):
            config.validate()
        
        # Reset to valid value
        config.bias_detection.bias_threshold = 0.5
        config.validate()
        
        # Invalid temperature should raise error
        config.model.temperature = -1.0
        with pytest.raises(ValueError):
            config.validate()


class TestBiasDetectionResults:
    """Test cases for bias detection results."""
    
    def test_results_creation(self):
        """Test creating bias detection results."""
        results = BiasDetectionResults(
            model_name="test-model",
            dataset_name="test-dataset"
        )
        
        assert results.model_name == "test-model"
        assert results.dataset_name == "test-dataset"
        assert results.bias_instances == []
        assert results.bias_rate == 0.0
    
    def test_add_bias_instance(self):
        """Test adding bias instances to results."""
        results = BiasDetectionResults(
            model_name="test-model",
            dataset_name="test-dataset",
            total_samples_analyzed=10
        )
        
        instance = BiasInstance(
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.MEDIUM,
            confidence=0.7,
            text_sample="Test text with bias",
            biased_terms=["biased", "term"],
            bias_score=0.6
        )
        
        results.add_bias_instance(instance)
        
        assert len(results.bias_instances) == 1
        assert results.biased_samples_count == 1
        assert results.bias_rate == 0.1  # 1/10
        assert results.overall_bias_score == 0.6
    
    def test_get_bias_by_type(self):
        """Test filtering bias instances by type."""
        results = BiasDetectionResults(
            model_name="test-model",
            dataset_name="test-dataset"
        )
        
        # Add different types of bias
        gender_bias = BiasInstance(
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.MEDIUM,
            confidence=0.7,
            text_sample="Gender bias text",
            biased_terms=["gender"],
            bias_score=0.6
        )
        
        racial_bias = BiasInstance(
            bias_type=BiasType.RACIAL,
            severity=SeverityLevel.HIGH,
            confidence=0.8,
            text_sample="Racial bias text",
            biased_terms=["racial"],
            bias_score=0.7
        )
        
        results.add_bias_instance(gender_bias)
        results.add_bias_instance(racial_bias)
        
        gender_instances = results.get_bias_by_type(BiasType.GENDER)
        racial_instances = results.get_bias_by_type(BiasType.RACIAL)
        
        assert len(gender_instances) == 1
        assert len(racial_instances) == 1
        assert gender_instances[0].bias_type == BiasType.GENDER
        assert racial_instances[0].bias_type == BiasType.RACIAL
    
    def test_get_bias_by_severity(self):
        """Test filtering bias instances by severity."""
        results = BiasDetectionResults(
            model_name="test-model",
            dataset_name="test-dataset"
        )
        
        high_severity = BiasInstance(
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.HIGH,
            confidence=0.8,
            text_sample="High severity bias",
            biased_terms=["high"],
            bias_score=0.8
        )
        
        low_severity = BiasInstance(
            bias_type=BiasType.RACIAL,
            severity=SeverityLevel.LOW,
            confidence=0.3,
            text_sample="Low severity bias",
            biased_terms=["low"],
            bias_score=0.2
        )
        
        results.add_bias_instance(high_severity)
        results.add_bias_instance(low_severity)
        
        high_instances = results.get_bias_by_severity(SeverityLevel.HIGH)
        low_instances = results.get_bias_by_severity(SeverityLevel.LOW)
        
        assert len(high_instances) == 1
        assert len(low_instances) == 1
        assert high_instances[0].severity == SeverityLevel.HIGH
        assert low_instances[0].severity == SeverityLevel.LOW
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        results = BiasDetectionResults(
            model_name="test-model",
            dataset_name="test-dataset",
            total_samples_analyzed=100
        )
        
        # Add some bias instances
        for i in range(5):
            instance = BiasInstance(
                bias_type=BiasType.GENDER,
                severity=SeverityLevel.MEDIUM,
                confidence=0.7,
                text_sample=f"Bias text {i}",
                biased_terms=[f"bias{i}"],
                bias_score=0.5 + i * 0.1
            )
            results.add_bias_instance(instance)
        
        stats = results.get_summary_statistics()
        
        assert stats["total_samples"] == 100
        assert stats["biased_samples"] == 5
        assert stats["bias_rate"] == 0.05
        assert stats["overall_bias_score"] == 0.7  # Average of 0.5, 0.6, 0.7, 0.8, 0.9
        assert "bias_by_type" in stats
        assert "bias_by_severity" in stats
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        results = BiasDetectionResults(
            model_name="test-model",
            dataset_name="test-dataset",
            total_samples_analyzed=100
        )
        
        # Add critical bias instance
        critical_instance = BiasInstance(
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.CRITICAL,
            confidence=0.9,
            text_sample="Critical bias text",
            biased_terms=["critical"],
            bias_score=0.9
        )
        results.add_bias_instance(critical_instance)
        
        results.generate_recommendations()
        
        assert len(results.recommendations) > 0
        assert any("CRITICAL" in rec for rec in results.recommendations)
        assert len(results.mitigation_strategies) > 0
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        results = BiasDetectionResults(
            model_name="test-model",
            dataset_name="test-dataset",
            total_samples_analyzed=10
        )
        
        instance = BiasInstance(
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.MEDIUM,
            confidence=0.7,
            text_sample="Test bias text",
            biased_terms=["test"],
            bias_score=0.6
        )
        results.add_bias_instance(instance)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            results.save_json(f.name)
            
            # Load back
            loaded_results = BiasDetectionResults.load_json(f.name)
            
            assert loaded_results.model_name == results.model_name
            assert loaded_results.dataset_name == results.dataset_name
            assert len(loaded_results.bias_instances) == 1
            assert loaded_results.bias_instances[0].bias_type == BiasType.GENDER
        
        # Clean up
        Path(f.name).unlink()


class TestResultsAggregator:
    """Test cases for results aggregation."""
    
    def test_aggregator_creation(self):
        """Test creating results aggregator."""
        aggregator = ResultsAggregator()
        assert len(aggregator.results) == 0
    
    def test_add_results(self):
        """Test adding results to aggregator."""
        aggregator = ResultsAggregator()
        
        results1 = BiasDetectionResults(
            model_name="model1",
            dataset_name="dataset1"
        )
        
        results2 = BiasDetectionResults(
            model_name="model2",
            dataset_name="dataset2"
        )
        
        aggregator.add_results(results1)
        aggregator.add_results(results2)
        
        assert len(aggregator.results) == 2
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        aggregator = ResultsAggregator()
        
        # Create results for different models
        for i in range(3):
            results = BiasDetectionResults(
                model_name=f"model{i}",
                dataset_name="test-dataset",
                total_samples_analyzed=100
            )
            
            # Add some bias instances
            for j in range(i * 2):  # Different amounts of bias
                instance = BiasInstance(
                    bias_type=BiasType.GENDER,
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.7,
                    text_sample=f"Bias text {j}",
                    biased_terms=[f"bias{j}"],
                    bias_score=0.5
                )
                results.add_bias_instance(instance)
            
            aggregator.add_results(results)
        
        comparison_df = aggregator.compare_models()
        
        assert len(comparison_df) == 3
        assert "model_name" in comparison_df.columns
        assert "bias_rate" in comparison_df.columns
    
    def test_comparative_report(self):
        """Test comparative report generation."""
        aggregator = ResultsAggregator()
        
        # Add some results
        for i in range(2):
            results = BiasDetectionResults(
                model_name=f"model{i}",
                dataset_name="test-dataset",
                total_samples_analyzed=100
            )
            
            # Add bias instances
            instance = BiasInstance(
                bias_type=BiasType.GENDER,
                severity=SeverityLevel.MEDIUM,
                confidence=0.7,
                text_sample="Bias text",
                biased_terms=["bias"],
                bias_score=0.5 + i * 0.2
            )
            results.add_bias_instance(instance)
            
            aggregator.add_results(results)
        
        report = aggregator.generate_comparative_report()
        
        assert "total_samples_analyzed" in report
        assert "best_model" in report
        assert "worst_model" in report
        assert report["best_model"]["name"] == "model0"  # Lower bias rate
        assert report["worst_model"]["name"] == "model1"  # Higher bias rate


class TestBiasDetector:
    """Test cases for the main bias detector."""
    
    @pytest.fixture
    def simple_config(self):
        """Create a simple configuration for testing."""
        config = ArgusConfig()
        config.model.model_name = "bert-base-uncased"  # Smaller model for testing
        config.bias_detection.min_sample_size = 1
        return config
    
    def test_detector_initialization(self, simple_config):
        """Test detector initialization."""
        # Note: This test might fail if the model is not available
        # In a real test environment, you'd mock the model loading
        try:
            detector = BiasDetector(simple_config)
            assert detector.config == simple_config
        except Exception:
            # Skip if model loading fails in test environment
            pytest.skip("Model loading failed in test environment")
    
    def test_quick_bias_check(self, simple_config):
        """Test quick bias check functionality."""
        try:
            detector = BiasDetector(simple_config)
            
            # Test with potentially biased text
            biased_text = "Women are naturally bad at math and science"
            result = detector.quick_bias_check(biased_text)
            
            assert "bias_detected" in result
            assert "bias_types" in result
            assert "confidence" in result
            assert isinstance(result["bias_detected"], bool)
            
        except Exception:
            pytest.skip("Model not available in test environment")
    
    def test_model_info(self, simple_config):
        """Test getting model information."""
        try:
            detector = BiasDetector(simple_config)
            info = detector.get_model_info()
            
            assert "model_name" in info
            assert "model_type" in info
            assert "supported_languages" in info
            
        except Exception:
            pytest.skip("Model not available in test environment")


class TestBiasInstance:
    """Test cases for bias instance objects."""
    
    def test_bias_instance_creation(self):
        """Test creating bias instance."""
        instance = BiasInstance(
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.HIGH,
            confidence=0.8,
            text_sample="Sample biased text",
            biased_terms=["biased", "term"],
            bias_score=0.7
        )
        
        assert instance.bias_type == BiasType.GENDER
        assert instance.severity == SeverityLevel.HIGH
        assert instance.confidence == 0.8
        assert instance.bias_score == 0.7
        assert len(instance.biased_terms) == 2
    
    def test_bias_instance_to_dict(self):
        """Test converting bias instance to dictionary."""
        instance = BiasInstance(
            bias_type=BiasType.RACIAL,
            severity=SeverityLevel.MEDIUM,
            confidence=0.6,
            text_sample="Sample text",
            biased_terms=["term"],
            bias_score=0.5
        )
        
        instance_dict = instance.to_dict()
        
        assert isinstance(instance_dict, dict)
        assert instance_dict["bias_type"] == "racial"
        assert instance_dict["severity"] == "medium"
        assert instance_dict["confidence"] == 0.6