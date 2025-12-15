from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from app.core.config import get_settings

client = None
collection = None

async def connect_db():
    global client, collection
    settings = get_settings()
    
    print(f"Connecting to MongoDB ({settings.environment} environment)...")
    print(f"MongoDB URL: {settings.mongodb_url[:30]}..." if len(settings.mongodb_url) > 30 else f"MongoDB URL: {settings.mongodb_url}")
    client = AsyncIOMotorClient(settings.mongodb_url)
    print("Connected to MongoDB.")
    db = client[settings.mongodb_db_name]
    collection = db[settings.mongodb_collection_name]

    return collection

async def close_db():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        print("MongoDB connection closed.")

async def log_inference(result: dict):
    global collection

    if collection is None:
        return
    
    try:
        document = {
            "label": result.get("label"),
            "confidence": result.get("confidence"),
            "timestamp": datetime.now()
        }
        await collection.insert_one(document)
    except Exception as e:
        print("DB logging failed:", e)
