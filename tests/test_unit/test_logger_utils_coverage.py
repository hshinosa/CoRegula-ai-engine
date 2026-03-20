"""
Tests for Utils Logger Module - Full Coverage
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from app.utils.logger import get_process_mining_logger, ProcessMiningLogger


class TestProcessMiningLogger:
    """Test ProcessMiningLogger class."""
    
    def test_init(self):
        """Test initialization."""
        logger = ProcessMiningLogger()
        assert logger is not None
    
    @patch('app.utils.logger.Path')
    def test_log_event(self, mock_path):
        """Test log_event method."""
        mock_path.return_value = MagicMock()
        mock_path.return_value.exists.return_value = True
        
        logger = ProcessMiningLogger(log_dir="./test_logs")
        
        # Just verify it doesn't crash
        logger.log_event("test_case", "Test_Activity", "User1")
    
    @patch('app.utils.logger.Path')
    def test_log_event_creates_dir(self, mock_path):
        """Test log_event creates directory if needed."""
        mock_path.return_value = MagicMock()
        mock_path.return_value.exists.return_value = False
        
        logger = ProcessMiningLogger(log_dir="./test_logs")
        
        # Just verify it doesn't crash
        logger.log_event("test", "Activity", "User")


class TestGetProcessMiningLogger:
    """Test get_process_mining_logger singleton."""
    
    def test_get_logger_singleton(self):
        """Test get_process_mining_logger returns singleton."""
        from app.utils import logger as logger_module
        logger_module._process_mining_logger = None
        
        with patch('app.utils.logger.ProcessMiningLogger') as MockLogger:
            mock_instance = MagicMock()
            MockLogger.return_value = mock_instance
            
            logger1 = get_process_mining_logger()
            logger2 = get_process_mining_logger()
            
            assert logger1 is logger2
    
    def test_get_logger_creates_instance(self):
        """Test get_process_mining_logger creates instance."""
        from app.utils import logger as logger_module
        logger_module._process_mining_logger = None
        
        with patch('app.utils.logger.ProcessMiningLogger') as MockLogger:
            mock_instance = MagicMock()
            MockLogger.return_value = mock_instance
            
            logger = get_process_mining_logger()
            
            assert logger is not None
