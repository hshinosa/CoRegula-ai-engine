import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.mongodb_logger import MongoDBLogger

@pytest.mark.asyncio
async def test_mongodb_logger_full():
    with patch('motor.motor_asyncio.AsyncIOMotorClient') as mock_client:
        logger = MongoDBLogger()
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.command = AsyncMock(return_value={'ok': 1.0})
        
        await logger.connect()
        assert logger._initialized is True
        
        # Mock log insertion
        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.insert_one = AsyncMock()
        
        await logger.log_activity({'test': 1})
        mock_coll.insert_one.assert_called()