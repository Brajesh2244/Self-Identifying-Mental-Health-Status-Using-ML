from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="HealthTech SaaS API",
    description="Enterprise API for AI-Powered Health Intelligence",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "operational", "service": "HealthTech API Core"}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
