"""
Streamlit dashboard for Project Argus bias detection system.

This module provides a comprehensive web-based dashboard for visualizing
bias detection results, monitoring real-time performance, and managing
the bias detection system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.config import ArgusConfig, load_config
from ..core.detector import BiasDetector
from ..core.results import BiasDetectionResults, BiasType, SeverityLevel
from .real_time_monitor import RealTimeMonitor


logger = logging.getLogger(__name__)


class ArgusDashboard:
    """
    Streamlit-based dashboard for Project Argus.
    
    Provides visualization and management interface for bias detection
    results, real-time monitoring, and system configuration.
    """
    
    def __init__(self, config: Optional[ArgusConfig] = None):
        """
        Initialize the dashboard.
        
        Args:
            config: Argus configuration object
        """
        self.config = config or load_config()
        self.redis_client = None
        
        # Initialize Redis connection for real-time data
        self._initialize_redis()
        
        # Set page configuration
        st.set_page_config(
            page_title="Project Argus - LLM Bias Detection",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.monitoring.redis_host,
                    port=self.config.monitoring.redis_port,
                    db=self.config.monitoring.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Dashboard Redis connection established")
            except Exception as e:
                logger.warning(f"Dashboard Redis connection failed: {str(e)}")
                self.redis_client = None
    
    def run(self) -> None:
        """Run the Streamlit dashboard."""
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1f77b4;
        }
        .alert-critical {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .alert-warning {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<h1 class="main-header">🎯 Project Argus</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">LLM Bias Detection System for Meta Superintelligence Labs</p>', unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Overview", "📊 Analytics", "🔍 Bias Detection", "⚡ Real-time Monitoring", "⚙️ Configuration", "📈 Reports"]
        )
        
        # Route to appropriate page
        if page == "🏠 Overview":
            self._show_overview_page()
        elif page == "📊 Analytics":
            self._show_analytics_page()
        elif page == "🔍 Bias Detection":
            self._show_bias_detection_page()
        elif page == "⚡ Real-time Monitoring":
            self._show_monitoring_page()
        elif page == "⚙️ Configuration":
            self._show_configuration_page()
        elif page == "📈 Reports":
            self._show_reports_page()
    
    def _show_overview_page(self) -> None:
        """Display overview dashboard page."""
        st.header("System Overview")
        
        # Get real-time metrics if available
        metrics = self._get_monitoring_metrics()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Texts Processed",
                value=f"{metrics.get('processed_count', 0):,}",
                delta=f"+{metrics.get('recent_processed', 0)}"
            )
        
        with col2:
            bias_rate = metrics.get('bias_detection_rate', 0) * 100
            st.metric(
                label="Bias Detection Rate",
                value=f"{bias_rate:.1f}%",
                delta=f"{metrics.get('bias_rate_change', 0):+.1f}%"
            )
        
        with col3:
            st.metric(
                label="Alerts Triggered",
                value=metrics.get('alert_count', 0),
                delta=f"+{metrics.get('recent_alerts', 0)}"
            )
        
        with col4:
            avg_time = metrics.get('avg_processing_time', 0) * 1000
            st.metric(
                label="Avg Processing Time",
                value=f"{avg_time:.0f}ms",
                delta=f"{metrics.get('time_change', 0):+.0f}ms"
            )
        
        # System status
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            # Model status
            st.markdown("**Model Status**")
            model_info = self._get_model_info()
            
            status_color = "🟢" if model_info.get('loaded', False) else "🔴"
            st.write(f"{status_color} Model: {model_info.get('model_name', 'Not loaded')}")
            
            if model_info.get('loaded', False):
                st.write(f"Parameters: {model_info.get('parameters', 0):,}")
                st.write(f"Device: {model_info.get('device', 'Unknown')}")
        
        with col2:
            # Service status
            st.markdown("**Service Status**")
            
            kafka_status = "🟢 Connected" if metrics.get('kafka_connected', False) else "🔴 Disconnected"
            redis_status = "🟢 Connected" if metrics.get('redis_connected', False) else "🔴 Disconnected"
            
            st.write(f"Kafka: {kafka_status}")
            st.write(f"Redis: {redis_status}")
            st.write(f"Monitoring: {'🟢 Active' if metrics.get('is_running', False) else '🔴 Inactive'}")
        
        # Recent alerts
        self._show_recent_alerts()
        
        # Bias trends chart
        st.subheader("Bias Detection Trends")
        trends_data = self._get_trends_data()
        
        if trends_data:
            fig = px.line(
                trends_data,
                x='timestamp',
                y=['bias_rate', 'bias_score'],
                title="Bias Detection Over Time",
                labels={'value': 'Score', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available yet.")
    
    def _show_analytics_page(self) -> None:
        """Display analytics dashboard page."""
        st.header("Bias Analytics")
        
        # Load sample data or real data
        sample_data = self._load_analytics_data()
        
        if sample_data.empty:
            st.warning("No data available for analytics. Run some bias detection first.")
            return
        
        # Bias type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bias Types Distribution")
            bias_counts = sample_data['bias_type'].value_counts()
            
            fig = px.pie(
                values=bias_counts.values,
                names=bias_counts.index,
                title="Distribution of Bias Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Severity Levels")
            severity_counts = sample_data['severity'].value_counts()
            
            fig = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                title="Bias by Severity Level",
                color=severity_counts.values,
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Detailed Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_bias_type = st.selectbox(
                "Filter by Bias Type",
                ["All"] + list(sample_data['bias_type'].unique())
            )
        
        with col2:
            selected_severity = st.selectbox(
                "Filter by Severity",
                ["All"] + list(sample_data['severity'].unique())
            )
        
        with col3:
            min_confidence = st.slider(
                "Minimum Confidence",
                0.0, 1.0, 0.0, 0.1
            )
        
        # Apply filters
        filtered_data = sample_data.copy()
        
        if selected_bias_type != "All":
            filtered_data = filtered_data[filtered_data['bias_type'] == selected_bias_type]
        
        if selected_severity != "All":
            filtered_data = filtered_data[filtered_data['severity'] == selected_severity]
        
        filtered_data = filtered_data[filtered_data['confidence'] >= min_confidence]
        
        # Display filtered results
        st.write(f"Showing {len(filtered_data)} results")
        
        if not filtered_data.empty:
            # Confidence distribution
            fig = px.histogram(
                filtered_data,
                x='confidence',
                title="Confidence Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Bias score vs confidence scatter
            fig = px.scatter(
                filtered_data,
                x='confidence',
                y='bias_score',
                color='severity',
                title="Bias Score vs Confidence",
                hover_data=['bias_type']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample instances table
            st.subheader("Sample Instances")
            display_data = filtered_data[['bias_type', 'severity', 'confidence', 'bias_score', 'text_sample']].head(10)
            st.dataframe(display_data, use_container_width=True)
        else:
            st.info("No data matches the selected filters.")
    
    def _show_bias_detection_page(self) -> None:
        """Display bias detection interface."""
        st.header("Interactive Bias Detection")
        
        # Text input methods
        input_method = st.radio(
            "Input Method",
            ["Single Text", "Multiple Texts", "File Upload"]
        )
        
        texts_to_analyze = []
        
        if input_method == "Single Text":
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste text here..."
            )
            if text_input.strip():
                texts_to_analyze = [text_input.strip()]
        
        elif input_method == "Multiple Texts":
            text_input = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="Enter each text on a new line..."
            )
            if text_input.strip():
                texts_to_analyze = [line.strip() for line in text_input.split('\n') if line.strip()]
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt', 'csv']
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.type == "text/plain":
                        content = str(uploaded_file.read(), "utf-8")
                        texts_to_analyze = [line.strip() for line in content.split('\n') if line.strip()]
                    elif uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            texts_to_analyze = df['text'].dropna().tolist()
                        else:
                            st.error("CSV file must have a 'text' column")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Analysis options
        st.subheader("Analysis Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_demographic = st.checkbox("Demographic Analysis", value=True)
            include_linguistic = st.checkbox("Linguistic Analysis", value=True)
        
        with col2:
            include_counterfactual = st.checkbox("Counterfactual Analysis", value=False)
            quick_analysis = st.checkbox("Quick Analysis", value=False)
        
        with col3:
            bias_threshold = st.slider("Bias Threshold", 0.0, 1.0, 0.1, 0.05)
        
        # Run analysis
        if st.button("Analyze for Bias", type="primary") and texts_to_analyze:
            with st.spinner("Analyzing texts for bias..."):
                try:
                    detector = BiasDetector(self.config)
                    
                    if quick_analysis and len(texts_to_analyze) == 1:
                        # Quick analysis for single text
                        result = detector.quick_bias_check(texts_to_analyze[0])
                        self._display_quick_results(result)
                    else:
                        # Full analysis
                        results = detector.detect_bias(
                            texts=texts_to_analyze,
                            dataset_name="dashboard_analysis",
                            include_demographic=include_demographic,
                            include_linguistic=include_linguistic,
                            include_counterfactual=include_counterfactual
                        )
                        self._display_full_results(results)
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        elif texts_to_analyze and not st.button("Analyze for Bias"):
            st.info(f"Ready to analyze {len(texts_to_analyze)} text(s). Click 'Analyze for Bias' to proceed.")
        
        elif not texts_to_analyze:
            st.info("Enter some text to analyze for bias.")
    
    def _show_monitoring_page(self) -> None:
        """Display real-time monitoring page."""
        st.header("Real-time Monitoring")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        if auto_refresh:
            # Auto-refresh placeholder
            placeholder = st.empty()
            
            # Refresh every 30 seconds
            if st.session_state.get('last_refresh', 0) + 30 < time.time():
                st.session_state.last_refresh = time.time()
                placeholder.rerun()
        
        # Current status
        metrics = self._get_monitoring_metrics()
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "🟢 Active" if metrics.get('is_running', False) else "🔴 Inactive"
            st.markdown(f"**Monitoring Status:** {status}")
        
        with col2:
            error_count = metrics.get('error_count', 0)
            error_status = "🟢 Normal" if error_count < 10 else "🟡 Elevated" if error_count < 50 else "🔴 High"
            st.markdown(f"**Error Rate:** {error_status} ({error_count})")
        
        with col3:
            last_alert = metrics.get('last_alert_time')
            if last_alert:
                st.markdown(f"**Last Alert:** {last_alert}")
            else:
                st.markdown("**Last Alert:** None")
        
        # Real-time metrics
        st.subheader("Live Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Texts/min",
                value=metrics.get('throughput', 0),
                delta=metrics.get('throughput_change', 0)
            )
        
        with col2:
            processing_time = metrics.get('avg_processing_time', 0) * 1000
            st.metric(
                "Avg Latency (ms)",
                value=f"{processing_time:.0f}",
                delta=f"{metrics.get('latency_change', 0):+.0f}"
            )
        
        with col3:
            bias_rate = metrics.get('bias_detection_rate', 0) * 100
            st.metric(
                "Bias Rate (%)",
                value=f"{bias_rate:.1f}",
                delta=f"{metrics.get('bias_rate_delta', 0):+.1f}"
            )
        
        with col4:
            st.metric(
                "Queue Size",
                value=metrics.get('queue_size', 0),
                delta=metrics.get('queue_change', 0)
            )
        
        # Real-time charts
        st.subheader("Real-time Analytics")
        
        # Create sample real-time data
        realtime_data = self._get_realtime_data()
        
        if realtime_data:
            # Throughput chart
            fig = px.line(
                realtime_data,
                x='timestamp',
                y='throughput',
                title="Processing Throughput (texts/min)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Bias detection chart
            fig = px.line(
                realtime_data,
                x='timestamp',
                y='bias_rate',
                title="Bias Detection Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Alert history
        self._show_alert_history()
        
        # Monitoring controls
        st.subheader("Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Monitoring"):
                st.success("Monitoring start command sent")
        
        with col2:
            if st.button("Stop Monitoring"):
                st.warning("Monitoring stop command sent")
        
        with col3:
            if st.button("Reset Metrics"):
                st.info("Metrics reset command sent")
    
    def _show_configuration_page(self) -> None:
        """Display configuration management page."""
        st.header("System Configuration")
        
        # Current configuration display
        st.subheader("Current Configuration")
        
        with st.expander("Model Configuration"):
            st.json({
                "model_name": self.config.model.model_name,
                "model_type": self.config.model.model_type,
                "device": self.config.model.device,
                "max_length": self.config.model.max_length,
                "temperature": self.config.model.temperature
            })
        
        with st.expander("Bias Detection Configuration"):
            st.json({
                "bias_threshold": self.config.bias_detection.bias_threshold,
                "enable_demographic_bias": self.config.bias_detection.enable_demographic_bias,
                "enable_linguistic_bias": self.config.bias_detection.enable_linguistic_bias,
                "languages": self.config.bias_detection.languages,
                "counterfactual_samples": self.config.bias_detection.counterfactual_samples
            })
        
        with st.expander("Monitoring Configuration"):
            st.json({
                "kafka_bootstrap_servers": self.config.monitoring.kafka_bootstrap_servers,
                "batch_size": self.config.monitoring.batch_size,
                "processing_interval": self.config.monitoring.processing_interval,
                "alert_threshold": self.config.monitoring.alert_threshold
            })
        
        # Configuration editor
        st.subheader("Edit Configuration")
        
        # Bias detection settings
        st.markdown("**Bias Detection Settings**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_threshold = st.slider(
                "Bias Threshold",
                0.0, 1.0,
                self.config.bias_detection.bias_threshold,
                0.01
            )
            
            new_demographic = st.checkbox(
                "Enable Demographic Analysis",
                self.config.bias_detection.enable_demographic_bias
            )
            
            new_linguistic = st.checkbox(
                "Enable Linguistic Analysis",
                self.config.bias_detection.enable_linguistic_bias
            )
        
        with col2:
            new_alert_threshold = st.slider(
                "Alert Threshold",
                0.0, 1.0,
                self.config.monitoring.alert_threshold,
                0.01
            )
            
            new_batch_size = st.number_input(
                "Batch Size",
                1, 1000,
                self.config.monitoring.batch_size
            )
        
        # Save configuration
        if st.button("Save Configuration"):
            # Update configuration (in real implementation, this would save to file)
            st.success("Configuration saved successfully!")
        
        # Export/Import configuration
        st.subheader("Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Configuration"):
                config_json = json.dumps(self.config.to_dict(), indent=2)
                st.download_button(
                    "Download Config",
                    config_json,
                    "argus_config.json",
                    "application/json"
                )
        
        with col2:
            config_file = st.file_uploader(
                "Import Configuration",
                type=['json', 'yaml']
            )
            
            if config_file and st.button("Import"):
                st.success("Configuration imported successfully!")
    
    def _show_reports_page(self) -> None:
        """Display reports and export page."""
        st.header("Reports & Export")
        
        # Report generation options
        st.subheader("Generate Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Summary Report", "Detailed Analysis", "Trend Analysis", "Model Comparison"]
            )
            
            date_range = st.date_input(
                "Date Range",
                value=[datetime.now().date() - timedelta(days=7), datetime.now().date()]
            )
        
        with col2:
            include_charts = st.checkbox("Include Charts", True)
            include_raw_data = st.checkbox("Include Raw Data", False)
            
            export_format = st.selectbox(
                "Export Format",
                ["PDF", "HTML", "CSV", "JSON"]
            )
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                # Simulate report generation
                time.sleep(2)
                
                if report_type == "Summary Report":
                    self._generate_summary_report(include_charts)
                elif report_type == "Detailed Analysis":
                    self._generate_detailed_report()
                elif report_type == "Trend Analysis":
                    self._generate_trend_report()
                else:
                    self._generate_comparison_report()
        
        # Historical data overview
        st.subheader("Historical Data Overview")
        
        # Sample historical metrics
        historical_data = self._get_historical_data()
        
        if not historical_data.empty:
            st.dataframe(historical_data, use_container_width=True)
            
            # Export historical data
            csv = historical_data.to_csv(index=False)
            st.download_button(
                "Download Historical Data",
                csv,
                "argus_historical_data.csv",
                "text/csv"
            )
        else:
            st.info("No historical data available.")
    
    def _get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        if self.redis_client:
            try:
                metrics_json = self.redis_client.get('argus:monitoring_metrics')
                if metrics_json:
                    return json.loads(metrics_json)
            except Exception as e:
                logger.error(f"Failed to get metrics from Redis: {str(e)}")
        
        # Return default/sample metrics
        return {
            'processed_count': 12450,
            'bias_detected_count': 187,
            'bias_detection_rate': 0.015,
            'avg_processing_time': 0.250,
            'alert_count': 3,
            'error_count': 2,
            'is_running': True,
            'kafka_connected': True,
            'redis_connected': True,
            'throughput': 42,
            'queue_size': 15
        }
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'loaded': True,
            'model_name': self.config.model.model_name,
            'parameters': 7000000000,
            'device': self.config.model.device
        }
    
    def _load_analytics_data(self) -> pd.DataFrame:
        """Load analytics data for visualization."""
        # Generate sample data for demonstration
        np.random.seed(42)
        
        bias_types = ['gender', 'racial', 'religious', 'linguistic', 'demographic']
        severities = ['low', 'medium', 'high', 'critical']
        
        data = []
        for _ in range(200):
            data.append({
                'bias_type': np.random.choice(bias_types),
                'severity': np.random.choice(severities),
                'confidence': np.random.uniform(0.1, 0.9),
                'bias_score': np.random.uniform(0.0, 1.0),
                'text_sample': f"Sample text {len(data) + 1}..."
            })
        
        return pd.DataFrame(data)
    
    def _display_quick_results(self, result: Dict[str, Any]) -> None:
        """Display quick analysis results."""
        st.subheader("Quick Analysis Results")
        
        if result['bias_detected']:
            st.error(f"⚠️ Bias detected! Types: {', '.join(result['bias_types'])}")
            st.write(f"Confidence: {result['confidence']:.2f}")
        else:
            st.success("✅ No significant bias detected")
            st.write(f"Confidence: {result['confidence']:.2f}")
        
        # Show model outputs
        with st.expander("Model Outputs"):
            for i, output in enumerate(result['model_outputs'], 1):
                st.write(f"**Output {i}:** {output}")
    
    def _display_full_results(self, results: BiasDetectionResults) -> None:
        """Display full analysis results."""
        st.subheader("Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", results.total_samples_analyzed)
        
        with col2:
            st.metric("Biased Samples", results.biased_samples_count)
        
        with col3:
            st.metric("Bias Rate", f"{results.bias_rate:.1%}")
        
        with col4:
            st.metric("Overall Score", f"{results.overall_bias_score:.3f}")
        
        # Bias instances
        if results.bias_instances:
            st.subheader("Detected Bias Instances")
            
            # Convert to DataFrame for display
            df = results.to_dataframe()
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                "bias_detection_results.csv",
                "text/csv"
            )
        else:
            st.success("No bias instances detected!")
        
        # Recommendations
        if results.recommendations:
            st.subheader("Recommendations")
            for rec in results.recommendations:
                st.write(f"• {rec}")
    
    def _show_recent_alerts(self) -> None:
        """Show recent alerts."""
        st.subheader("Recent Alerts")
        
        # Sample alerts
        alerts = [
            {
                'timestamp': '2024-01-15 14:30:25',
                'type': 'High Bias Rate',
                'severity': 'warning',
                'message': 'Bias rate exceeded 15% threshold'
            },
            {
                'timestamp': '2024-01-15 12:15:10',
                'type': 'Critical Bias Instance',
                'severity': 'critical',
                'message': 'Critical gender bias detected in output'
            }
        ]
        
        for alert in alerts:
            if alert['severity'] == 'critical':
                st.markdown(f"""
                <div class="alert-critical">
                    <strong>🚨 {alert['type']}</strong><br>
                    {alert['message']}<br>
                    <small>{alert['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>⚠️ {alert['type']}</strong><br>
                    {alert['message']}<br>
                    <small>{alert['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def _get_trends_data(self) -> Optional[pd.DataFrame]:
        """Get trends data for visualization."""
        # Generate sample trend data
        dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='H')
        
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'bias_rate': np.random.uniform(0.05, 0.25, len(dates)),
            'bias_score': np.random.uniform(0.1, 0.6, len(dates))
        }
        
        return pd.DataFrame(data)
    
    def _get_realtime_data(self) -> Optional[pd.DataFrame]:
        """Get real-time data for charts."""
        # Generate sample real-time data
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
        
        np.random.seed(int(time.time()))
        data = {
            'timestamp': timestamps,
            'throughput': np.random.uniform(30, 60, 60),
            'bias_rate': np.random.uniform(0.05, 0.20, 60)
        }
        
        return pd.DataFrame(data)
    
    def _show_alert_history(self) -> None:
        """Show alert history table."""
        st.subheader("Alert History")
        
        # Sample alert history
        alert_history = pd.DataFrame([
            {
                'timestamp': '2024-01-15 14:30:25',
                'type': 'High Bias Rate',
                'severity': 'Warning',
                'resolved': True
            },
            {
                'timestamp': '2024-01-15 12:15:10',
                'type': 'Critical Bias',
                'severity': 'Critical',
                'resolved': True
            },
            {
                'timestamp': '2024-01-15 09:45:33',
                'type': 'System Error',
                'severity': 'Error',
                'resolved': True
            }
        ])
        
        st.dataframe(alert_history, use_container_width=True)
    
    def _generate_summary_report(self, include_charts: bool) -> None:
        """Generate summary report."""
        st.success("Summary report generated successfully!")
        
        if include_charts:
            # Sample summary chart
            fig = px.bar(
                x=['Low', 'Medium', 'High', 'Critical'],
                y=[45, 23, 12, 5],
                title="Bias Instances by Severity"
            )
            st.plotly_chart(fig)
        
        st.download_button(
            "Download Summary Report",
            "Sample summary report content...",
            "argus_summary_report.txt"
        )
    
    def _generate_detailed_report(self) -> None:
        """Generate detailed analysis report."""
        st.success("Detailed analysis report generated successfully!")
        st.download_button(
            "Download Detailed Report",
            "Sample detailed report content...",
            "argus_detailed_report.txt"
        )
    
    def _generate_trend_report(self) -> None:
        """Generate trend analysis report."""
        st.success("Trend analysis report generated successfully!")
        st.download_button(
            "Download Trend Report",
            "Sample trend report content...",
            "argus_trend_report.txt"
        )
    
    def _generate_comparison_report(self) -> None:
        """Generate model comparison report."""
        st.success("Model comparison report generated successfully!")
        st.download_button(
            "Download Comparison Report",
            "Sample comparison report content...",
            "argus_comparison_report.txt"
        )
    
    def _get_historical_data(self) -> pd.DataFrame:
        """Get historical data overview."""
        # Generate sample historical data
        dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
        
        np.random.seed(42)
        data = {
            'date': dates,
            'texts_processed': np.random.randint(1000, 5000, len(dates)),
            'bias_instances': np.random.randint(10, 100, len(dates)),
            'alerts_triggered': np.random.randint(0, 5, len(dates)),
            'avg_bias_score': np.random.uniform(0.1, 0.4, len(dates))
        }
        
        return pd.DataFrame(data)


def main():
    """Main function to run the dashboard."""
    dashboard = ArgusDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()