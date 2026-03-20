"""
Tests for MongoDB Logger Service - Full Coverage
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO
from app.services.mongodb_logger import (
    MongoDBLogger,
    get_mongo_logger,
)


class TestMongoDBLoggerFull:
    """Test MongoDBLogger full coverage."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock MongoDB client."""
        mock = MagicMock()
        mock.admin.command = AsyncMock()
        
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.create_index = AsyncMock()
        mock_collection.insert_one = AsyncMock()
        mock_collection.find = MagicMock(return_value=MagicMock(
            sort=MagicMock(return_value=MagicMock(
                limit=MagicMock(return_value=MagicMock(
                    to_list=AsyncMock(return_value=[{
                        "_id": "mock_id",
                        "CaseID": "test",
                        "Activity": "Test",
                        "Timestamp": datetime.now(),
                        "Resource": "User",
                        "Attributes": {
                            "original_text": "Test",
                            "srl_object": "Object",
                            "educational_category": "Cognitive",
                            "is_hot": True,
                            "lexical_variety": 0.5,
                            "scaffolding_trigger": False
                        }
                    }])
                ))
            ))
        ))
        
        mock.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.activity_logs = mock_collection
        mock_db.silence_events = mock_collection
        
        return mock
    
    @pytest.fixture
    def mongo_logger(self):
        """Create MongoDBLogger instance."""
        return MongoDBLogger()
    
    def test_init(self, mongo_logger):
        """Test initialization."""
        assert mongo_logger.client is None
        assert mongo_logger.db is None
    
    @pytest.mark.asyncio
    async def test_connect_disabled(self, mongo_logger):
        """Test connect when disabled."""
        with patch('app.services.mongodb_logger.settings') as mock_settings:
            mock_settings.ENABLE_MONGODB_LOGGING = False
            
            await mongo_logger.connect()
            
            assert mongo_logger.enabled is False
            assert mongo_logger.client is None
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mongo_logger, mock_client):
        """Test connect with success."""
        with patch('app.services.mongodb_logger.settings') as mock_settings:
            mock_settings.ENABLE_MONGODB_LOGGING = True
            mock_settings.MONGO_URI = 'mongodb://localhost'
            mock_settings.MONGO_DB_NAME = 'test_db'
            
            with patch('app.services.mongodb_logger.AsyncIOMotorClient', return_value=mock_client):
                await mongo_logger.connect()
                
                assert mongo_logger.client is not None
                assert mongo_logger.db is not None
                assert mongo_logger.enabled is True
    
    @pytest.mark.asyncio
    async def test_connect_exception(self, mongo_logger):
        """Test connect with exception."""
        with patch('app.services.mongodb_logger.settings') as mock_settings:
            mock_settings.ENABLE_MONGODB_LOGGING = True
            mock_settings.MONGO_URI = 'mongodb://localhost'
            mock_settings.MONGO_DB_NAME = 'test_db'
            
            mock_client = MagicMock()
            mock_client.admin.command = AsyncMock(side_effect=Exception("Connection failed"))
            
            with patch('app.services.mongodb_logger.AsyncIOMotorClient', return_value=mock_client):
                await mongo_logger.connect()
                
                assert mongo_logger.enabled is False
    
    @pytest.mark.asyncio
    async def test_log_activity_disabled(self, mongo_logger):
        """Test log_activity when disabled."""
        mongo_logger.enabled = False
        
        entry = {"CaseID": "test", "Activity": "Test"}
        await mongo_logger.log_activity(entry)
        
        # Should not raise and should not log
    
    @pytest.mark.asyncio
    async def test_log_activity_no_db(self, mongo_logger):
        """Test log_activity without db."""
        mongo_logger.enabled = True
        mongo_logger.db = None
        
        entry = {"CaseID": "test", "Activity": "Test"}
        await mongo_logger.log_activity(entry)
    
    @pytest.mark.asyncio
    async def test_log_activity_success(self, mongo_logger, mock_client):
        """Test log_activity with success."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        entry = {"CaseID": "test", "Activity": "Test"}
        await mongo_logger.log_activity(entry)
        
        mock_client['test_db'].activity_logs.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_activity_adds_timestamp(self, mongo_logger, mock_client):
        """Test log_activity adds timestamp."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        entry = {"CaseID": "test", "Activity": "Test"}
        await mongo_logger.log_activity(entry)
        
        assert "Timestamp" in entry
    
    @pytest.mark.asyncio
    async def test_log_activity_exception(self, mongo_logger, mock_client):
        """Test log_activity with exception."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        mock_client['test_db'].activity_logs.insert_one = AsyncMock(side_effect=Exception("Error"))
        
        entry = {"CaseID": "test", "Activity": "Test"}
        
        # Should not raise
        await mongo_logger.log_activity(entry)
    
    @pytest.mark.asyncio
    async def test_log_intervention(self, mongo_logger, mock_client):
        """Test log_intervention method."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        await mongo_logger.log_intervention(
            group_id="group1",
            intervention_type="redirect",
            reason="Test reason",
            metadata={"key": "value"}
        )
        
        mock_client['test_db'].activity_logs.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_intervention_custom_session(self, mongo_logger, mock_client):
        """Test log_intervention with custom session."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        await mongo_logger.log_intervention(
            group_id="group1",
            intervention_type="silence",
            reason="Test",
            metadata={},
            session_id="5"
        )
        
        # Should include custom session_id
        call_args = mock_client['test_db'].activity_logs.insert_one.call_args
        assert "session_5" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_get_activity_logs_disabled(self, mongo_logger):
        """Test get_activity_logs when disabled."""
        mongo_logger.enabled = False
        
        result = await mongo_logger.get_activity_logs(case_id="test")
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_activity_logs_no_db(self, mongo_logger):
        """Test get_activity_logs without db."""
        mongo_logger.enabled = True
        mongo_logger.db = None
        
        result = await mongo_logger.get_activity_logs(case_id="test")
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_activity_logs_success(self, mongo_logger, mock_client):
        """Test get_activity_logs with success."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        result = await mongo_logger.get_activity_logs(case_id="test")
        
        assert len(result) == 1
        assert result[0]["CaseID"] == "test"
    
    @pytest.mark.asyncio
    async def test_get_activity_logs_with_resource(self, mongo_logger, mock_client):
        """Test get_activity_logs with resource filter."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        result = await mongo_logger.get_activity_logs(resource="User")
        
        assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_get_activity_logs_chat_space_id(self, mongo_logger, mock_client):
        """Test get_activity_logs with chat_space_id."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        result = await mongo_logger.get_activity_logs(chat_space_id="chat1")
        
        assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_get_activity_logs_timestamp_conversion(self, mongo_logger, mock_client):
        """Test get_activity_logs converts datetime to ISO format."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        # Mock with datetime object
        mock_dt = datetime(2024, 1, 1, 12, 0, 0)
        mock_client['test_db'].activity_logs.find = MagicMock(return_value=MagicMock(
            sort=MagicMock(return_value=MagicMock(
                limit=MagicMock(return_value=MagicMock(
                    to_list=AsyncMock(return_value=[{
                        "_id": "mock_id",
                        "Timestamp": mock_dt,
                        "CaseID": "test",
                        "Activity": "Test",
                        "Resource": "User",
                        "Attributes": {}
                    }])
                ))
            ))
        ))
        
        result = await mongo_logger.get_activity_logs(case_id="test")
        
        assert len(result) == 1
        assert result[0]["Timestamp"] == mock_dt.isoformat()
    
    @pytest.mark.asyncio
    async def test_get_activity_logs_exception(self, mongo_logger, mock_client):
        """Test get_activity_logs with exception."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        mock_client['test_db'].activity_logs.find = MagicMock(side_effect=Exception("Error"))
        
        result = await mongo_logger.get_activity_logs(case_id="test")
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_export_to_csv(self, mongo_logger, mock_client):
        """Test export_to_csv method."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        
        result = await mongo_logger.export_to_csv(case_id="test")
        
        assert isinstance(result, str)
        assert "CaseID" in result
    
    @pytest.mark.asyncio
    async def test_export_to_csv_no_logs(self, mongo_logger, mock_client):
        """Test export_to_csv with no logs."""
        mongo_logger.enabled = True
        mongo_logger.db = mock_client['test_db']
        mock_client['test_db'].activity_logs.find = MagicMock(return_value=MagicMock(
            sort=MagicMock(return_value=MagicMock(
                limit=MagicMock(return_value=MagicMock(
                    to_list=AsyncMock(return_value=[])
                ))
            ))
        ))
        
        result = await mongo_logger.export_to_csv(case_id="test")
        
        assert isinstance(result, str)
        assert "CaseID" in result  # Header should still be present
    
    @pytest.mark.asyncio
    async def test_close(self, mongo_logger, mock_client):
        """Test close method."""
        mongo_logger.client = mock_client
        
        await mongo_logger.close()
        
        mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_no_client(self, mongo_logger):
        """Test close without client."""
        mongo_logger.client = None
        
        # Should not raise
        await mongo_logger.close()


class TestGetMongoLoggerFull:
    """Test get_mongo_logger singleton full coverage."""
    
    def test_singleton_creates_instance(self):
        """Test singleton creates new instance."""
        from app.services import mongodb_logger
        mongodb_logger._mongo_logger = None
        
        logger = get_mongo_logger()
        
        assert logger is not None
        assert isinstance(logger, MongoDBLogger)
    
    def test_singleton_returns_existing(self):
        """Test singleton returns existing instance."""
        from app.services import mongodb_logger
        
        existing = MagicMock()
        mongodb_logger._mongo_logger = existing
        
        result = get_mongo_logger()
        
        assert result is existing
