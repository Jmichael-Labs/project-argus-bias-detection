"""
Results handling and reporting for Project Argus bias detection system.

This module provides comprehensive result management including storage,
analysis, visualization, and reporting capabilities.
"""

import json
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum


class BiasType(Enum):
    """Enumeration of bias types detected by Argus."""
    DEMOGRAPHIC = "demographic"
    LINGUISTIC = "linguistic"
    CULTURAL = "cultural"
    GENDER = "gender"
    RACIAL = "racial"
    RELIGIOUS = "religious"
    SOCIOECONOMIC = "socioeconomic"
    AGE = "age"
    DISABILITY = "disability"


class SeverityLevel(Enum):
    """Bias severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasInstance:
    """Individual bias detection instance."""
    
    bias_type: BiasType
    severity: SeverityLevel
    confidence: float
    text_sample: str
    biased_terms: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical measures
    bias_score: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0
    
    # Counterfactual analysis
    counterfactual_text: Optional[str] = None
    counterfactual_score: Optional[float] = None
    
    # Metadata
    model_name: str = ""
    language: str = "en"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bias instance to dictionary."""
        result = asdict(self)
        result['bias_type'] = self.bias_type.value
        result['severity'] = self.severity.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class DemographicAnalysis:
    """Analysis results for demographic bias."""
    
    group_comparisons: Dict[str, Dict[str, float]]
    parity_metrics: Dict[str, float]
    disparate_impact: float
    statistical_significance: bool
    protected_attributes: List[str]
    
    def get_worst_disparity(self) -> Tuple[str, float]:
        """Get the worst disparity found."""
        max_disparity = 0.0
        worst_group = ""
        
        for group, metrics in self.group_comparisons.items():
            if "disparity" in metrics and metrics["disparity"] > max_disparity:
                max_disparity = metrics["disparity"]
                worst_group = group
        
        return worst_group, max_disparity


@dataclass
class LinguisticAnalysis:
    """Analysis results for linguistic bias."""
    
    language_distribution: Dict[str, float]
    dialect_bias: Dict[str, float]
    formality_bias: Dict[str, float]
    vocabulary_bias: Dict[str, List[str]]
    sentiment_disparities: Dict[str, float]


@dataclass
class CounterfactualAnalysis:
    """Analysis results from counterfactual generation."""
    
    original_outputs: List[str]
    counterfactual_outputs: List[str]
    bias_differences: List[float]
    average_bias_change: float
    significant_changes: int
    
    def get_effectiveness_score(self) -> float:
        """Calculate effectiveness of counterfactual interventions."""
        if not self.bias_differences:
            return 0.0
        
        positive_changes = sum(1 for diff in self.bias_differences if diff > 0.1)
        return positive_changes / len(self.bias_differences)


@dataclass
class BiasDetectionResults:
    """Comprehensive bias detection results."""
    
    # Basic information
    model_name: str
    dataset_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    detection_duration: float = 0.0
    
    # Individual bias instances
    bias_instances: List[BiasInstance] = field(default_factory=list)
    
    # Aggregate analyses
    demographic_analysis: Optional[DemographicAnalysis] = None
    linguistic_analysis: Optional[LinguisticAnalysis] = None
    counterfactual_analysis: Optional[CounterfactualAnalysis] = None
    
    # Overall metrics
    overall_bias_score: float = 0.0
    total_samples_analyzed: int = 0
    biased_samples_count: int = 0
    bias_rate: float = 0.0
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    def add_bias_instance(self, instance: BiasInstance) -> None:
        """Add a bias instance to the results."""
        self.bias_instances.append(instance)
        self.biased_samples_count += 1
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update aggregate metrics based on bias instances."""
        if self.total_samples_analyzed > 0:
            self.bias_rate = self.biased_samples_count / self.total_samples_analyzed
        
        if self.bias_instances:
            self.overall_bias_score = np.mean([inst.bias_score for inst in self.bias_instances])
    
    def get_bias_by_type(self, bias_type: BiasType) -> List[BiasInstance]:
        """Get all bias instances of a specific type."""
        return [inst for inst in self.bias_instances if inst.bias_type == bias_type]
    
    def get_bias_by_severity(self, severity: SeverityLevel) -> List[BiasInstance]:
        """Get all bias instances of a specific severity."""
        return [inst for inst in self.bias_instances if inst.severity == severity]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the detection results."""
        bias_by_type = {}
        bias_by_severity = {}
        
        for bias_type in BiasType:
            count = len(self.get_bias_by_type(bias_type))
            bias_by_type[bias_type.value] = count
        
        for severity in SeverityLevel:
            count = len(self.get_bias_by_severity(severity))
            bias_by_severity[severity.value] = count
        
        return {
            "total_samples": self.total_samples_analyzed,
            "biased_samples": self.biased_samples_count,
            "bias_rate": self.bias_rate,
            "overall_bias_score": self.overall_bias_score,
            "bias_by_type": bias_by_type,
            "bias_by_severity": bias_by_severity,
            "detection_duration": self.detection_duration,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def generate_recommendations(self) -> None:
        """Generate recommendations based on detected biases."""
        self.recommendations.clear()
        self.mitigation_strategies.clear()
        
        # Check overall bias rate
        if self.bias_rate > 0.2:
            self.recommendations.append(
                "High bias rate detected. Consider comprehensive model retraining with bias mitigation techniques."
            )
        elif self.bias_rate > 0.1:
            self.recommendations.append(
                "Moderate bias rate detected. Implement targeted bias reduction strategies."
            )
        
        # Check specific bias types
        demographic_bias = len(self.get_bias_by_type(BiasType.DEMOGRAPHIC))
        if demographic_bias > 0:
            self.recommendations.append(
                f"Demographic bias detected in {demographic_bias} instances. "
                "Review training data for representation issues."
            )
            self.mitigation_strategies.append("Implement demographic parity constraints during training")
        
        gender_bias = len(self.get_bias_by_type(BiasType.GENDER))
        if gender_bias > 0:
            self.recommendations.append(
                f"Gender bias detected in {gender_bias} instances. "
                "Apply gender-neutral language processing."
            )
            self.mitigation_strategies.append("Use counterfactual data augmentation for gender terms")
        
        # Check severity levels
        critical_bias = len(self.get_bias_by_severity(SeverityLevel.CRITICAL))
        if critical_bias > 0:
            self.recommendations.append(
                f"CRITICAL: {critical_bias} instances of critical bias detected. "
                "Immediate intervention required before deployment."
            )
        
        # Linguistic bias
        if self.linguistic_analysis and self.linguistic_analysis.sentiment_disparities:
            max_disparity = max(self.linguistic_analysis.sentiment_disparities.values())
            if max_disparity > 0.3:
                self.recommendations.append(
                    "Significant linguistic bias detected. Consider multilingual training data balancing."
                )
                self.mitigation_strategies.append("Implement cross-lingual bias reduction techniques")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert bias instances to pandas DataFrame."""
        if not self.bias_instances:
            return pd.DataFrame()
        
        data = []
        for instance in self.bias_instances:
            row = {
                "bias_type": instance.bias_type.value,
                "severity": instance.severity.value,
                "confidence": instance.confidence,
                "bias_score": instance.bias_score,
                "p_value": instance.p_value,
                "effect_size": instance.effect_size,
                "text_sample": instance.text_sample[:100] + "..." if len(instance.text_sample) > 100 else instance.text_sample,
                "biased_terms": ", ".join(instance.biased_terms),
                "model_name": instance.model_name,
                "language": instance.language,
                "timestamp": instance.timestamp,
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "detection_duration": self.detection_duration,
            "bias_instances": [inst.to_dict() for inst in self.bias_instances],
            "summary_statistics": self.get_summary_statistics(),
            "recommendations": self.recommendations,
            "mitigation_strategies": self.mitigation_strategies,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_csv(self, filepath: Union[str, Path]) -> None:
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    def save_pickle(self, filepath: Union[str, Path]) -> None:
        """Save results to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> "BiasDetectionResults":
        """Load results from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct bias instances
        bias_instances = []
        for inst_data in data.get("bias_instances", []):
            instance = BiasInstance(
                bias_type=BiasType(inst_data["bias_type"]),
                severity=SeverityLevel(inst_data["severity"]),
                confidence=inst_data["confidence"],
                text_sample=inst_data["text_sample"],
                biased_terms=inst_data["biased_terms"],
                context=inst_data.get("context", {}),
                bias_score=inst_data.get("bias_score", 0.0),
                p_value=inst_data.get("p_value", 1.0),
                effect_size=inst_data.get("effect_size", 0.0),
                counterfactual_text=inst_data.get("counterfactual_text"),
                counterfactual_score=inst_data.get("counterfactual_score"),
                model_name=inst_data.get("model_name", ""),
                language=inst_data.get("language", "en"),
                timestamp=datetime.fromisoformat(inst_data["timestamp"]),
            )
            bias_instances.append(instance)
        
        # Create results object
        results = cls(
            model_name=data["model_name"],
            dataset_name=data["dataset_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            detection_duration=data.get("detection_duration", 0.0),
            bias_instances=bias_instances,
            recommendations=data.get("recommendations", []),
            mitigation_strategies=data.get("mitigation_strategies", []),
        )
        
        # Update metrics
        results.total_samples_analyzed = data["summary_statistics"]["total_samples"]
        results._update_metrics()
        
        return results
    
    @classmethod
    def load_pickle(cls, filepath: Union[str, Path]) -> "BiasDetectionResults":
        """Load results from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ResultsAggregator:
    """Aggregates and compares multiple bias detection results."""
    
    def __init__(self):
        self.results: List[BiasDetectionResults] = []
    
    def add_results(self, results: BiasDetectionResults) -> None:
        """Add results to the aggregator."""
        self.results.append(results)
    
    def compare_models(self) -> pd.DataFrame:
        """Compare bias metrics across different models."""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in self.results:
            summary = result.get_summary_statistics()
            comparison_data.append(summary)
        
        return pd.DataFrame(comparison_data)
    
    def get_bias_trends(self) -> Dict[str, List[float]]:
        """Get bias trends over time."""
        trends = {
            "timestamps": [],
            "bias_rates": [],
            "bias_scores": [],
        }
        
        # Sort results by timestamp
        sorted_results = sorted(self.results, key=lambda x: x.timestamp)
        
        for result in sorted_results:
            trends["timestamps"].append(result.timestamp)
            trends["bias_rates"].append(result.bias_rate)
            trends["bias_scores"].append(result.overall_bias_score)
        
        return trends
    
    def generate_comparative_report(self) -> Dict[str, Any]:
        """Generate a comprehensive comparative report."""
        if not self.results:
            return {}
        
        # Calculate aggregate statistics
        total_samples = sum(r.total_samples_analyzed for r in self.results)
        total_biased = sum(r.biased_samples_count for r in self.results)
        average_bias_rate = total_biased / total_samples if total_samples > 0 else 0
        
        # Find best and worst performing models
        best_model = min(self.results, key=lambda x: x.bias_rate)
        worst_model = max(self.results, key=lambda x: x.bias_rate)
        
        return {
            "total_samples_analyzed": total_samples,
            "total_biased_samples": total_biased,
            "average_bias_rate": average_bias_rate,
            "best_model": {
                "name": best_model.model_name,
                "bias_rate": best_model.bias_rate,
                "bias_score": best_model.overall_bias_score,
            },
            "worst_model": {
                "name": worst_model.model_name,
                "bias_rate": worst_model.bias_rate,
                "bias_score": worst_model.overall_bias_score,
            },
            "model_comparison": self.compare_models().to_dict(),
            "bias_trends": self.get_bias_trends(),
        }