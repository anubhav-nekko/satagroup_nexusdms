"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.config import API_V1_PREFIX, PROJECT_NAME, API_VERSION
from app.auth.router import router as auth_router
from app.jds.router import router as jds_router
from app.cvs.router import router as cvs_router
from app.analysis.router import router as analysis_router

# Create FastAPI app
app = FastAPI(
    title=PROJECT_NAME,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(jds_router)
app.include_router(cvs_router)
app.include_router(analysis_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": PROJECT_NAME,
        "version": API_VERSION,
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=PROJECT_NAME,
        version=API_VERSION,
        description="API for skill analysis between job descriptions and CVs",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Add security requirement to all operations
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if "security" not in operation:
                operation["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
