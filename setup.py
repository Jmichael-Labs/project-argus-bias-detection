"""
Project Argus: Personal LLM Bias Detection Demonstration
Advanced bias detection and mitigation framework for large language models.
"""

from setuptools import setup, find_packages
import os

# Read the README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open(os.path.join(current_dir, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="project-argus-bias-detection",
    version="1.0.0",
    author="Michael Jaramillo",
    author_email="jmichaeloficial@gmail.com",
    description="Personal demonstration of advanced LLM bias detection capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meta-labs/project-argus-bias-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "cupy-cuda12x>=12.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "argus-detect=argus.cli:main",
            "argus-monitor=argus.monitors.cli:main",
            "argus-dashboard=argus.monitors.dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "argus": ["configs/*.yaml", "data/*.json", "models/*.pkl"],
    },
    keywords=[
        "bias detection",
        "llm",
        "fairness",
        "meta",
        "artificial intelligence",
        "machine learning",
        "nlp",
        "transformer models",
        "ethical ai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/meta-labs/project-argus-bias-detection/issues",
        "Source": "https://github.com/meta-labs/project-argus-bias-detection",
        "Documentation": "https://project-argus-docs.meta.com",
        "LinkedIn": "https://www.linkedin.com/in/michael-jaramillo-b61815278",
    },
)