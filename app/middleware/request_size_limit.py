"""
Request Size Limit Middleware
=============================
KOL-148: Limit request body size to prevent DoS attacks.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class LimitRequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.
    
    Prevents memory exhaustion attacks by rejecting oversized request bodies.
    """
    
    def __init__(self, app, max_size_bytes: int = 10 * 1024 * 1024):
        """
        Initialize middleware with size limit.
        
        Args:
            app: The ASGI application
            max_size_bytes: Maximum allowed request body size (default 10MB)
        """
        super().__init__(app)
        self.max_size_bytes = max_size_bytes
    
    async def dispatch(self, request: Request, call_next):
        """
        Check request body size and reject if too large.
        
        Args:
            request: The incoming request
            call_next: Function to call the next middleware/endpoint
            
        Returns:
            Response from next handler or 413 error response
        """
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            if len(body) > self.max_size_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": "PAYLOAD_TOO_LARGE",
                        "message": f"Request body exceeds maximum size of {self.max_size_bytes} bytes"
                    }
                )
        
        return await call_next(request)
