"""
Project Argus: LLM Bias Detection System for Meta Superintelligence Labs

A comprehensive framework for detecting, analyzing, and mitigating bias in large language models,
specifically optimized for Meta's LLM ecosystem including Llama models.

Author: Michael Jaramillo
Contact: jmichaeloficial@gmail.com
LinkedIn: https://www.linkedin.com/in/michael-jaramillo-b61815278
"""

__version__ = "1.0.0"
__author__ = "Michael Jaramillo"
__email__ = "jmichaeloficial@gmail.com"
__license__ = "MIT"

from .core.detector import BiasDetector
from .core.config import ArgusConfig
from .core.results import BiasDetectionResults

__all__ = [
    "BiasDetector",
    "ArgusConfig", 
    "BiasDetectionResults",
    "__version__",
    "__author__",
    "__email__",
]