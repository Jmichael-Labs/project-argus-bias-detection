"""
Real-time monitoring system for Project Argus.

This module implements Kafka-based streaming bias detection and monitoring
for production environments, providing real-time alerts and analytics.
"""

import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import asdict
import uuid

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import NoBrokersAvailable, KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("Kafka libraries not available. Real-time monitoring will be limited.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Caching will be limited.")

from ..core.config import ArgusConfig
from ..core.detector import BiasDetector
from ..core.results import BiasDetectionResults, BiasInstance


logger = logging.getLogger(__name__)


class RealTimeMonitor:
    """
    Real-time bias monitoring system using Kafka streams.
    
    This monitor processes incoming text streams, performs bias detection,
    and sends alerts when bias thresholds are exceeded.
    """
    
    def __init__(self, config: ArgusConfig):
        """
        Initialize the real-time monitor.
        
        Args:
            config: Argus configuration object
        """
        self.config = config
        self.is_running = False
        self.consumer = None
        self.producer = None
        self.redis_client = None
        
        # Initialize bias detector
        self.detector = BiasDetector(config)
        
        # Monitoring state
        self.processed_count = 0
        self.bias_detected_count = 0
        self.alert_count = 0
        self.last_alert_time = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Performance metrics
        self.processing_times = []
        self.error_count = 0
        
        # Initialize connections
        self._initialize_kafka()
        self._initialize_redis()
    
    def _initialize_kafka(self) -> None:
        """Initialize Kafka consumer and producer."""
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available. Real-time monitoring disabled.")
            return
        
        try:
            # Initialize consumer
            self.consumer = KafkaConsumer(
                self.config.monitoring.kafka_topic_input,
                bootstrap_servers=self.config.monitoring.kafka_bootstrap_servers,
                group_id=self.config.monitoring.kafka_consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000
            )
            
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.monitoring.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=3,
                acks='all'
            )
            
            logger.info("Kafka connections initialized successfully")
            
        except NoBrokersAvailable:
            logger.error("No Kafka brokers available. Check Kafka configuration.")
            self.consumer = None
            self.producer = None
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {str(e)}")
            self.consumer = None
            self.producer = None
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection for caching and state management."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Caching disabled.")
            return
        
        try:
            self.redis_client = redis.Redis(
                host=self.config.monitoring.redis_host,
                port=self.config.monitoring.redis_port,
                db=self.config.monitoring.redis_db,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            self.redis_client = None
    
    def start_monitoring(self) -> None:
        """Start the real-time monitoring process."""
        if not self.consumer:
            logger.error("Cannot start monitoring: Kafka consumer not available")
            return
        
        logger.info("Starting Project Argus real-time bias monitoring")
        self.is_running = True
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # Start metrics reporting thread
        metrics_thread = threading.Thread(target=self._metrics_reporting_loop, daemon=True)
        metrics_thread.start()
        
        logger.info("Real-time monitoring started successfully")
    
    def stop_monitoring(self) -> None:
        """Stop the real-time monitoring process."""
        logger.info("Stopping real-time monitoring")
        self.is_running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that processes Kafka messages."""
        batch = []
        batch_start_time = time.time()
        
        while self.is_running:
            try:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    # Process accumulated batch if timeout reached
                    if batch and (time.time() - batch_start_time) > self.config.monitoring.processing_interval:
                        self._process_batch(batch)
                        batch = []
                        batch_start_time = time.time()
                    continue
                
                # Process messages
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Parse message
                            data = message.value
                            
                            # Validate message format
                            if not self._validate_message(data):
                                logger.warning(f"Invalid message format: {data}")
                                continue
                            
                            # Add to batch
                            batch.append(data)
                            
                            # Process batch if it reaches target size
                            if len(batch) >= self.config.monitoring.batch_size:
                                self._process_batch(batch)
                                batch = []
                                batch_start_time = time.time()
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                            self.error_count += 1
                            continue
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                self.error_count += 1
                time.sleep(1)  # Brief pause before retrying
        
        # Process any remaining batch
        if batch:
            self._process_batch(batch)
    
    def _validate_message(self, data: Dict[str, Any]) -> bool:
        """Validate incoming message format."""
        required_fields = ['text', 'timestamp']
        return all(field in data for field in required_fields)
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of messages for bias detection."""
        start_time = time.time()
        
        try:
            # Extract texts from batch
            texts = [item['text'] for item in batch]
            
            # Perform bias detection
            results = self.detector.detect_bias(
                texts=texts,
                dataset_name=f"realtime_batch_{int(start_time)}",
                include_demographic=True,
                include_linguistic=True,
                include_counterfactual=False  # Skip for performance in real-time
            )
            
            # Process results
            self._process_results(results, batch)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.processed_count += len(batch)
            
            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
            
            logger.info(f"Processed batch of {len(batch)} texts in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            self.error_count += 1
    
    def _process_results(self, results: BiasDetectionResults, batch: List[Dict[str, Any]]) -> None:
        """Process bias detection results and send alerts if needed."""
        # Check for high-severity bias
        critical_instances = [
            inst for inst in results.bias_instances 
            if inst.severity.value in ['high', 'critical']
        ]
        
        if critical_instances:
            self.bias_detected_count += len(critical_instances)
            
            # Check if alert threshold is exceeded
            if results.bias_rate > self.config.monitoring.alert_threshold:
                self._send_alert(results, critical_instances)
        
        # Send results to output topic
        self._send_results(results, batch)
        
        # Cache results if Redis is available
        if self.redis_client:
            self._cache_results(results)
    
    def _send_alert(self, results: BiasDetectionResults, critical_instances: List[BiasInstance]) -> None:
        """Send bias alert."""
        current_time = datetime.now()
        
        # Rate limiting: don't send alerts too frequently
        if (self.last_alert_time and 
            current_time - self.last_alert_time < timedelta(minutes=5)):
            return
        
        alert_data = {
            'alert_id': str(uuid.uuid4()),
            'timestamp': current_time.isoformat(),
            'alert_type': 'bias_detected',
            'severity': 'high',
            'bias_rate': results.bias_rate,
            'bias_score': results.overall_bias_score,
            'critical_instances_count': len(critical_instances),
            'model_name': results.model_name,
            'sample_instances': [
                {
                    'bias_type': inst.bias_type.value,
                    'severity': inst.severity.value,
                    'confidence': inst.confidence,
                    'text_sample': inst.text_sample[:200] + '...' if len(inst.text_sample) > 200 else inst.text_sample
                }
                for inst in critical_instances[:3]  # Sample of critical instances
            ]
        }
        
        # Send to Kafka alert topic
        if self.producer:
            try:
                self.producer.send('argus-alerts', value=alert_data)
                self.producer.flush()
                logger.warning(f"BIAS ALERT: {len(critical_instances)} critical instances detected")
            except Exception as e:
                logger.error(f"Failed to send alert to Kafka: {str(e)}")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {str(e)}")
        
        self.last_alert_time = current_time
        self.alert_count += 1
    
    def _send_results(self, results: BiasDetectionResults, batch: List[Dict[str, Any]]) -> None:
        """Send results to output Kafka topic."""
        if not self.producer:
            return
        
        try:
            # Prepare output message
            output_data = {
                'batch_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'model_name': results.model_name,
                'processed_count': len(batch),
                'bias_rate': results.bias_rate,
                'overall_bias_score': results.overall_bias_score,
                'bias_instances_count': len(results.bias_instances),
                'bias_summary': results.get_summary_statistics(),
                'processing_duration': results.detection_duration
            }
            
            # Send to output topic
            self.producer.send(
                self.config.monitoring.kafka_topic_output,
                value=output_data
            )
            
            # Don't flush immediately for performance
            
        except Exception as e:
            logger.error(f"Failed to send results to Kafka: {str(e)}")
    
    def _cache_results(self, results: BiasDetectionResults) -> None:
        """Cache results in Redis for dashboard access."""
        if not self.redis_client:
            return
        
        try:
            # Cache summary statistics
            cache_key = f"argus:results:{int(time.time())}"
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'bias_rate': results.bias_rate,
                'bias_score': results.overall_bias_score,
                'instances_count': len(results.bias_instances),
                'model_name': results.model_name
            }
            
            # Store with expiration (24 hours)
            self.redis_client.setex(cache_key, 86400, json.dumps(cache_data))
            
            # Update running totals
            self.redis_client.incr('argus:total_processed', len(results.bias_instances))
            self.redis_client.incr('argus:total_biased', results.biased_samples_count)
            
        except Exception as e:
            logger.error(f"Failed to cache results: {str(e)}")
    
    def _metrics_reporting_loop(self) -> None:
        """Report monitoring metrics periodically."""
        while self.is_running:
            try:
                time.sleep(self.config.monitoring.processing_interval)
                
                # Calculate metrics
                avg_processing_time = (
                    sum(self.processing_times) / len(self.processing_times)
                    if self.processing_times else 0
                )
                
                bias_detection_rate = (
                    self.bias_detected_count / self.processed_count
                    if self.processed_count > 0 else 0
                )
                
                # Log metrics
                logger.info(
                    f"Monitoring metrics - "
                    f"Processed: {self.processed_count}, "
                    f"Bias detected: {self.bias_detected_count}, "
                    f"Bias rate: {bias_detection_rate:.3f}, "
                    f"Avg processing time: {avg_processing_time:.3f}s, "
                    f"Alerts sent: {self.alert_count}, "
                    f"Errors: {self.error_count}"
                )
                
                # Cache metrics if Redis is available
                if self.redis_client:
                    metrics = {
                        'processed_count': self.processed_count,
                        'bias_detected_count': self.bias_detected_count,
                        'bias_detection_rate': bias_detection_rate,
                        'avg_processing_time': avg_processing_time,
                        'alert_count': self.alert_count,
                        'error_count': self.error_count,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.redis_client.setex(
                        'argus:monitoring_metrics',
                        300,  # 5 minutes expiration
                        json.dumps(metrics)
                    )
                
            except Exception as e:
                logger.error(f"Error in metrics reporting: {str(e)}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'is_running': self.is_running,
            'processed_count': self.processed_count,
            'bias_detected_count': self.bias_detected_count,
            'bias_detection_rate': (
                self.bias_detected_count / self.processed_count
                if self.processed_count > 0 else 0
            ),
            'alert_count': self.alert_count,
            'error_count': self.error_count,
            'last_alert_time': self.last_alert_time.isoformat() if self.last_alert_time else None,
            'avg_processing_time': (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0
            ),
            'kafka_connected': self.consumer is not None and self.producer is not None,
            'redis_connected': self.redis_client is not None
        }
    
    def simulate_input_stream(self, texts: List[str], delay: float = 1.0) -> None:
        """
        Simulate input stream for testing purposes.
        
        Args:
            texts: List of texts to simulate as stream
            delay: Delay between messages in seconds
        """
        if not self.producer:
            logger.error("Cannot simulate stream: Kafka producer not available")
            return
        
        logger.info(f"Starting stream simulation with {len(texts)} texts")
        
        for i, text in enumerate(texts):
            try:
                message = {
                    'text': text,
                    'timestamp': datetime.now().isoformat(),
                    'message_id': str(uuid.uuid4()),
                    'source': 'simulation'
                }
                
                self.producer.send(
                    self.config.monitoring.kafka_topic_input,
                    value=message
                )
                
                logger.debug(f"Sent message {i+1}/{len(texts)}")
                
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to send simulated message: {str(e)}")
        
        self.producer.flush()
        logger.info("Stream simulation completed")


class AlertManager:
    """Manages alert delivery and escalation."""
    
    def __init__(self, config: ArgusConfig):
        self.config = config
        self.alert_history = []
        
    def send_email_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send email alert (placeholder implementation)."""
        logger.info(f"EMAIL ALERT: {alert_data['alert_type']} - {alert_data['severity']}")
        
    def send_slack_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send Slack alert (placeholder implementation)."""
        logger.info(f"SLACK ALERT: {alert_data['alert_type']} - {alert_data['severity']}")
        
    def send_webhook_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send webhook alert (placeholder implementation)."""
        logger.info(f"WEBHOOK ALERT: {alert_data['alert_type']} - {alert_data['severity']}")


def create_monitor(config_path: Optional[str] = None) -> RealTimeMonitor:
    """
    Create and configure a real-time monitor.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configured RealTimeMonitor instance
    """
    from ..core.config import load_config
    
    config = load_config(config_path)
    return RealTimeMonitor(config)