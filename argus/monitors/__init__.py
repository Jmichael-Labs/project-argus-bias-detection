"""
Monitoring modules for Project Argus bias detection system.

This package contains real-time monitoring and dashboard components:
- RealTimeMonitor: Kafka-based real-time bias monitoring
- Dashboard: Streamlit-based visualization dashboard
"""

from .real_time_monitor import RealTimeMonitor
from .dashboard import ArgusDashboard

__all__ = ["RealTimeMonitor", "ArgusDashboard"]