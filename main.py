

import uvicorn

if __name__ == "__main__":
    #import app from generation.py
    uvicorn.run("generation:app", host="0.0.0.0", port=8001, reload=True)
    
    