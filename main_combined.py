import os
import json
import base64
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from google import genai
from google.genai import types

# Import database schema components
from database import init_db, get_db, ImageEmbedding

# --- 1. Lifespan Context Handler (Kept Untouched) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        print("Database initialized successfully on startup.")
    except Exception as e:
        print(f"Warning: Could not initialize database on startup. Error: {e}")
    yield

# Initialize the FastAPI app linking the lifespan engine
app = FastAPI(
    title="ImgVerifynStore Orchestration Service",
    description="Unified API to embed an image, check duplicate vectors, and conditionally store them.",
    lifespan=lifespan
)

# Initialize the Gemini SDK Client Lazily/Globally
client = None
try:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true" or "GOOGLE_CLOUD_PROJECT" in os.environ:
        client = genai.Client(vertexai=True)
    else:
        client = genai.Client()
except Exception as e:
    print(f"Warning: Gemini client initialization deferred. Error: {e}")


# --- 2. Resilient Pydantic Schemas for Strict Input/Output Parsing ---

class MetadataSchema(BaseModel):
    case_id: Optional[str] = None
    part_category: Optional[str] = None
    dealer_id: Optional[str] = None
    uploaded_by: Optional[str] = None
    
    model_config = {
        "extra": "allow"
    }

class ImgVerifyStoreRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded string of the claim image")
    metadata: Any = Field(..., description="Flexible metadata wrapper to handle Pega serialization overrides")
    similarity_threshold: Optional[Any] = Field(0.85, description="Cosine similarity cut-off flag")

    # This validator intercepts the metadata field and fixes Pega stringification on the fly
    @field_validator('metadata', mode='before')
    @classmethod
    def sanitize_metadata(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                # If passed as a string representation, parse it back to a clean dictionary
                return json.loads(value)
            except Exception:
                return {}
        return value

class ImgVerifyStoreResponse(BaseModel):
    is_duplicate: bool
    status_message: str
    vector_id: str
    similarity_score: float
    matching_metadata: Dict[str, Any]


# --- 3. Helper Function: Call Gemini Embedding Pipeline ---
def get_gemini_embedding(image_base64: str) -> List[float]:
    global client
    try:
        if client is None:
            if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true" or "GOOGLE_CLOUD_PROJECT" in os.environ:
                client = genai.Client(vertexai=True)
            else:
                client = genai.Client()
                
        base64_data = image_base64
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
            
        image_bytes = base64.b64decode(base64_data)
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )
        
        result = client.models.embed_content(
            model='gemini-embedding-2',
            contents=[image_part],
            config=types.EmbedContentConfig(output_dimensionality=1408)
        )
        
        if result.embeddings and len(result.embeddings) > 0:
            return result.embeddings[0].values
        else:
            raise HTTPException(status_code=500, detail="Gemini model returned an empty vector list.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini processing failure: {str(e)}")


# --- 4. Main Service Endpoint API ---

@app.post("/img_verify_and_store", response_model=ImgVerifyStoreResponse)
async def img_verify_and_store(request: ImgVerifyStoreRequest, db: Session = Depends(get_db)):
    try:
        # A. Clean up similarity threshold validation strings safely
        try:
            if request.similarity_threshold is None or str(request.similarity_threshold).strip() == "":
                threshold = 0.85
            else:
                threshold = float(str(request.similarity_threshold).strip())
        except Exception:
            threshold = 0.85

        # B. Convert metadata back into a structural schema object for scanning
        raw_metadata = request.metadata if isinstance(request.metadata, dict) else {}
        metadata_obj = MetadataSchema(**raw_metadata)

        # Step 1: Core Base64 -> Vector transformation via Gemini API
        generated_vector = get_gemini_embedding(request.image_base64)
        
        # Step 2: Build Database query to check vector distances
        query = db.query(
            ImageEmbedding,
            (1 - ImageEmbedding.embedding.cosine_distance(generated_vector)).label("similarity")
        )
        
        # Apply metadata filtering constraints safely ignoring None and empty Pega strings
        filter_dict = metadata_obj.model_dump(exclude_none=True)
        for key, value in filter_dict.items():
            if value and str(value).strip() != "":
                if key != "case_id": # Check matching parts across other cases, ignoring itself
                    query = query.filter(ImageEmbedding.metadata_.contains({key: value}))
                
        # Filter against the similarity threshold
        query = query.filter(
            (1 - ImageEmbedding.embedding.cosine_distance(generated_vector)) >= threshold
        )
        
        # Pull the absolute closest record
        top_match = query.order_by(ImageEmbedding.embedding.cosine_distance(generated_vector)).first()
        
        # Step 3: Branching Logic based on structural duplicate checks
        if top_match:
            db_record, similarity_score = top_match
            return ImgVerifyStoreResponse(
                is_duplicate=True,
                status_message="Duplicate image detected in vector repository.",
                vector_id=str(db_record.id),
                similarity_score=float(similarity_score),
                matching_metadata=db_record.metadata_
            )
        else:
            new_embedding = ImageEmbedding(
                embedding=generated_vector,
                metadata_=metadata_obj.model_dump()
            )
            db.add(new_embedding)
            db.commit()
            db.refresh(new_embedding)
            
            return ImgVerifyStoreResponse(
                is_duplicate=False,
                status_message="Image is unique. Successfully stored in vector store.",
                vector_id=str(new_embedding.id),
                similarity_score=0.0,
                matching_metadata={}
            )
            
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unified service execution crashed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)