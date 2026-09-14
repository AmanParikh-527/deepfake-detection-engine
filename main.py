"""Root entrypoint for single-deploy and local execution of TrueSight AI.

Runs the FastAPI backend and serves all static frontend pages from one origin.
"""

import os
import uvicorn
from api.index import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting TrueSight AI on http://localhost:{port}")
    uvicorn.run("api.index:app", host="0.0.0.0", port=port, reload=True)
