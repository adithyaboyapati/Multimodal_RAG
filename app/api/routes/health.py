"""
Health probes for container orchestration (Kubernetes / Docker).
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_app_settings
from app.config.settings import Settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", status_code=status.HTTP_200_OK)
def liveness_probe():
    """Liveness probe indicating the service process is up."""
    return {
        "status": "live",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
def readiness_probe(settings: Settings = Depends(get_app_settings)):
    """Readiness probe verifying operational configuration."""
    has_groq = bool(settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here")
    has_pinecone = bool(settings.pinecone_api_key and settings.pinecone_api_key != "your_pinecone_api_key_here")

    if not has_groq or not has_pinecone:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "degraded",
                "groq_configured": has_groq,
                "pinecone_configured": has_pinecone,
            },
        )

    return {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "groq": "configured",
            "pinecone": "configured",
            "index_name": settings.pinecone_index_name,
            "namespace": settings.pinecone_namespace,
        },
    }
