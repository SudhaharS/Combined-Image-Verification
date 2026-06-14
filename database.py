import os
from sqlalchemy import create_engine, Column, String, JSON, DateTime, func, cast
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from pgvector.sqlalchemy import Vector

# Database Configuration
# Default to a local postgres or a cloud sql proxy for dev if not set
#DATABASE_URL = os.environ.get(
 #   "DATABASE_URL", 
  #  "postgresql://user:password@localhost:5432/vector_db"
#)

# Build connection dynamically from your structured environment variables
DB_USER = os.environ.get("DB_USER", "<<>>"))
DB_PASS = os.environ.get("DB_PASS", "<<>>"))
DB_NAME = os.environ.get("DB_NAME", "<<>>"))
DB_HOST = os.environ.get("DB_HOST", "<<>>"))

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@/{DB_NAME}?host={DB_HOST}"

# Create the SQLAlchemy engine
# pool_pre_ping ensures connections aren't stale
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    # Using UUID as primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # The vector dimension is 1408 (output of Gemini multimodal embedding)
    embedding = Column(Vector(1408))
    
    # Metadata as JSONB to allow flexible filtering
    metadata_ = Column("metadata", JSONB)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    # Make sure pgvector extension is enabled before creating tables
    with engine.connect() as conn:
        conn.execute(func.text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Create tables
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
