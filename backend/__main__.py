import uvicorn
import os
from app import create_app, load_model

if __name__ == "__main__":
    
    print("\nStarting server...")
    app = create_app()

    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000))
    )