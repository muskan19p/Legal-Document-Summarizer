from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.requests import Request

app = FastAPI()

# Allow cross-origin requests (CORS) for frontend (you can replace "*" with your React app's URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your React container origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine to load HTML files
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Endpoint to accept document upload (for summarization)
@app.post("/upload-file/")
async def upload_file(file: str):
    # Process file (save, parse, or analyze)
    # Placeholder for document processing logic, e.g., converting PDF to text or summarization
    return {"message": "File uploaded successfully", "file_name": file}

# Example route for rating (you could connect this to the frontend's stars rating section)
@app.post("/submit-rating/")
async def submit_rating(rating: int):
    # Save the rating or do something with it
    return {"message": f"Rating of {rating} received successfully"}

# Example route for additional feedback
@app.post("/submit-feedback/")
async def submit_feedback(feedback: str):
    # Process feedback (e.g., save it to a database)
    return {"message": "Feedback received successfully", "feedback": feedback}

# Root endpoint for testing (returns a simple JSON message)
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}
