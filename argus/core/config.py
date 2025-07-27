"""
Configuration management for Project Argus bias detection system.

This module handles all configuration aspects including model settings,
bias detection parameters, and system configurations.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for language models being analyzed."""
    
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    model_type: str = "llama"  # llama, gpt, bert, etc.
    device: str = "auto"  # auto, cpu, cuda, mps
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_cache: bool = True
    trust_remote_code: bool = False


@dataclass
class BiasDetectionConfig:
    """Configuration for bias detection algorithms."""
    
    # Detection methods to use
    enable_demographic_bias: bool = True
    enable_linguistic_bias: bool = True
    enable_cultural_bias: bool = True
    enable_gender_bias: bool = True
    enable_racial_bias: bool = True
    enable_religious_bias: bool = True
    enable_socioeconomic_bias: bool = True
    
    # Detection thresholds
    bias_threshold: float = 0.1
    significance_level: float = 0.05
    min_sample_size: int = 100
    
    # Counterfactual generation
    counterfactual_samples: int = 1000
    perturbation_rate: float = 0.1
    
    # Multi-language support
    languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de", "zh", "ar"])
    detect_language: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring."""
    
    # Kafka settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_input: str = "argus-input"
    kafka_topic_output: str = "argus-results"
    kafka_consumer_group: str = "argus-bias-detector"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Monitoring intervals
    batch_size: int = 100
    processing_interval: int = 60  # seconds
    alert_threshold: float = 0.2
    
    # Dashboard settings
    dashboard_port: int = 8501
    dashboard_host: str = "0.0.0.0"
    refresh_interval: int = 30


@dataclass
class DataConfig:
    """Configuration for data handling and storage."""
    
    # Data paths
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    cache_dir: str = "cache"
    
    # Database settings
    database_url: Optional[str] = None
    use_database: bool = False
    
    # Data processing
    max_text_length: int = 2048
    preprocessing_enabled: bool = True
    anonymization_enabled: bool = True


@dataclass
class ArgusConfig:
    """Main configuration class for Project Argus."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    bias_detection: BiasDetectionConfig = field(default_factory=BiasDetectionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # System settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    debug_mode: bool = False
    random_seed: int = 42
    
    # Meta-specific settings
    meta_api_key: Optional[str] = None
    meta_model_hub: str = "https://huggingface.co/meta-llama"
    use_meta_optimizations: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ArgusConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ArgusConfig":
        """Create configuration from dictionary."""
        # Extract nested configurations
        model_config = ModelConfig(**config_dict.get("model", {}))
        bias_config = BiasDetectionConfig(**config_dict.get("bias_detection", {}))
        monitoring_config = MonitoringConfig(**config_dict.get("monitoring", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        # Extract main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["model", "bias_detection", "monitoring", "data"]}
        
        return cls(
            model=model_config,
            bias_detection=bias_config,
            monitoring=monitoring_config,
            data=data_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "bias_detection": self.bias_detection.__dict__,
            "monitoring": self.monitoring.__dict__,
            "data": self.data.__dict__,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "debug_mode": self.debug_mode,
            "random_seed": self.random_seed,
            "meta_api_key": self.meta_api_key,
            "meta_model_hub": self.meta_model_hub,
            "use_meta_optimizations": self.use_meta_optimizations,
        }
    
    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Model settings
        if os.getenv("ARGUS_MODEL_NAME"):
            self.model.model_name = os.getenv("ARGUS_MODEL_NAME")
        if os.getenv("ARGUS_DEVICE"):
            self.model.device = os.getenv("ARGUS_DEVICE")
        
        # Monitoring settings
        if os.getenv("ARGUS_KAFKA_SERVERS"):
            self.monitoring.kafka_bootstrap_servers = os.getenv("ARGUS_KAFKA_SERVERS")
        if os.getenv("ARGUS_REDIS_HOST"):
            self.monitoring.redis_host = os.getenv("ARGUS_REDIS_HOST")
        
        # Meta settings
        if os.getenv("META_API_KEY"):
            self.meta_api_key = os.getenv("META_API_KEY")
        
        # System settings
        if os.getenv("ARGUS_LOG_LEVEL"):
            self.log_level = os.getenv("ARGUS_LOG_LEVEL")
        if os.getenv("ARGUS_DEBUG"):
            self.debug_mode = os.getenv("ARGUS_DEBUG").lower() == "true"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate thresholds
        if not 0 <= self.bias_detection.bias_threshold <= 1:
            raise ValueError("bias_threshold must be between 0 and 1")
        
        if not 0 < self.bias_detection.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
        
        # Validate model settings
        if self.model.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        if not 0 <= self.model.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        # Validate monitoring settings
        if self.monitoring.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.monitoring.processing_interval <= 0:
            raise ValueError("processing_interval must be positive")


def load_config(config_path: Optional[Union[str, Path]] = None) -> ArgusConfig:
    """
    Load Argus configuration from file or create default.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
    
    Returns:
        ArgusConfig: Loaded or default configuration
    """
    if config_path is None:
        # Create default configuration
        config = ArgusConfig()
    else:
        config = ArgusConfig.from_yaml(config_path)
    
    # Update from environment variables
    config.update_from_env()
    
    # Validate configuration
    config.validate()
    
    return config


# Default configuration paths
DEFAULT_CONFIG_PATH = Path("configs/bias_detection.yaml")
META_CONFIG_PATH = Path("configs/meta_llm_config.yaml")