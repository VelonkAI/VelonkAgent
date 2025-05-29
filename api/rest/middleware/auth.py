"""
Enterprise Authentication Middleware - JWT/OAuth2/API Key Auth with Zero-Trust Security
"""

from __future__ import annotations
import base64
import hashlib
import hmac
import os
import re
import time
from typing import Awaitable, Callable, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp, Receive, Scope, Send

from core.config import SecurityConfig
from utils.logger import Logger
from utils.metrics import MetricsSystem
from kafka.producer import KafkaProducer
from redis.cluster import RedisCluster

#region Security Models

class JWTClaims(BaseModel):
    sub: str  # Subject (user/agent ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration
    iat: int  # Issued at
    jti: str  # JWT ID
    roles: list[str]
    permissions: list[str]
    mfa: Optional[bool] = False

class SecurityContext(BaseModel):
    identity: str
    auth_method: str
    session_id: Optional[str]
    device_fingerprint: Optional[str]
    location: Optional[Tuple[float, float]]

class AuditLogEntry(BaseModel):
    timestamp: float
    event_type: str  # auth_success, auth_failure, token_revoked
    identity: Optional[str]
    source_ip: str
    user_agent: str
    resource: str
    metadata: dict

#endregion

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        public_routes: list[str] = [],
        security_config: SecurityConfig = SecurityConfig(),
        redis: RedisCluster = RedisCluster.from_url(os.getenv("REDIS_URL")),
        kafka: KafkaProducer = KafkaProducer.get_instance()
    ):
        super().__init__(app)
        self.security_config = security_config
        self.redis = redis
        self.kafka = kafka
        self.logger = Logger.get_instance()
        self.metrics = MetricsSystem.get_instance()
        self.public_routes = [
            re.compile(pattern) for pattern in public_routes
        ]
        self.oauth2_scheme = OAuth2PasswordBearer(
            tokenUrl=f"{security_config.issuer}/oauth2/token",
            auto_error=False
        )
        self._public_keys = self._fetch_jwks()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> JSONResponse:
        # Skip authentication for public routes
        if self._is_public_route(request.url.path):
            return await call_next(request)

        try:
            # Phase 1: Request Analysis
            security_context = await self._analyze_request(request)

            # Phase 2: Threat Detection
            await self._detect_threats(request, security_context)

            # Phase 3: Authentication
            auth_result = await self._authenticate(request)

            # Phase 4: Authorization
            await self._authorize(request, auth_result)

            # Phase 5: Context Injection
            request.state.security = security_context
            request.state.auth = auth_result

            # Phase 6: Response Processing
            response = await call_next(request)

            # Phase 7: Post-Processing
            await self._apply_security_headers(response)
            self._log_audit_event(request, "auth_success")

            return response

        except HTTPException as e:
            return self._handle_auth_error(request, e)
        except Exception as e:
            return self._handle_unexpected_error(request, e)

    #region Core Authentication Methods

    async def _authenticate(self, request: Request) -> JWTClaims:
        # Try multiple authentication methods
        auth_methods = [
            self._jwt_auth,
            self._api_key_auth,
            self._oauth2_auth,
            self._cookie_auth
        ]

        for method in auth_methods:
            result = await method(request)
            if result:
                return result

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No valid authentication method found",
            headers={"WWW-Authenticate": "Bearer"}
        )

    async def _jwt_auth(self, request: Request) -> Optional[JWTClaims]:
        token = await self._extract_jwt(request)
        if not token:
            return None

        # Check token blacklist
        if await self.redis.exists(f"jwt_blacklist:{token}"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token revoked",
                headers={"WWW-Authenticate": "Bearer"}
            )

        try:
            payload = jwt.decode(
                token,
                self._get_public_key(token),
                algorithms=[self.security_config.jwt_algorithm],
                issuer=self.security_config.issuer,
                audience=self.security_config.audience,
                options={
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_iat": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_sub": True,
                    "verify_jti": True,
                    "leeway": 30
                }
            )
            return JWTClaims(**payload)
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"JWT validation failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )

    async def _api_key_auth(self, request: Request) -> Optional[JWTClaims]:
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return None

        # Constant-time comparison
        if not hmac.compare_digest(
            self.security_config.api_key_hash,
            hashlib.sha512(api_key.encode()).hexdigest()
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        return JWTClaims(
            sub="system",
            iss=self.security_config.issuer,
            aud=self.security_config.audience,
            exp=int(time.time()) + 3600,
            iat=int(time.time()),
            jti="api_key_auth",
            roles=["service_account"],
            permissions=["*"],
            mfa=False
        )

    #endregion

    #region Security Operations

    async def _detect_threats(self, request: Request, context: SecurityContext):
        # Rate limiting
        rate_key = f"rate_limit:{context.identity}:{request.url.path}"
        current = await self.redis.incr(rate_key)
        if current == 1:
            await self.redis.expire(rate_key, 60)
        
        if current > self.security_config.rate_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

        # Check IP reputation
        if await self.redis.sismember("blacklisted_ips", context.identity):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Blocked IP address"
            )

    async def _authorize(self, request: Request, claims: JWTClaims):
        required_permissions = self._get_required_permissions(request)
        
        # Check permissions
        if "*" not in claims.permissions:
            missing = [
                perm for perm in required_permissions
                if perm not in claims.permissions
            ]
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing permissions: {', '.join(missing)}"
                )

        # Check MFA
        if self.security_config.mfa_required and not claims.mfa:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Multi-factor authentication required"
            )

    #endregion

    #region Utility Methods

    def _is_public_route(self, path: str) -> bool:
        return any(pattern.fullmatch(path) for pattern in self.public_routes)

    async def _extract_jwt(self, request: Request) -> Optional[str]:
        locations = [
            request.headers.get("Authorization"),
            request.cookies.get("access_token"),
            request.query_params.get("token")
        ]

        for token in locations:
            if token:
                scheme, param = get_authorization_scheme_param(token)
                if scheme.lower() == "bearer":
                    return param
        return None

    def _get_public_key(self, token: str) -> dict:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        return self._public_keys.get(kid)

    def _fetch_jwks(self) -> dict:
        # Cached JWKS fetching with TTL
        cache_key = "jwks_cache"
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        jwks_url = f"{self.security_config.issuer}/.well-known/jwks.json"
        response = requests.get(jwks_url, timeout=5)
        jwks = response.json()
        self.redis.setex(cache_key, 3600, json.dumps(jwks))
        return {key["kid"]: key for key in jwks["keys"]}

    #endregion

    #region Security Headers

    def _apply_security_headers(self, response: JSONResponse):
        headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "accelerometer=(), camera=(), geolocation=()"
        }
        for header, value in headers.items():
            response.headers[header] = value

    #endregion

    #region Error Handling

    def _handle_auth_error(self, request: Request, exc: HTTPException):
        self._log_audit_event(request, "auth_failure", exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
            headers=exc.headers
        )

    def _handle_unexpected_error(self, request: Request, exc: Exception):
        self.logger.error(f"Auth error: {str(exc)}", exc_info=exc)
        self._log_audit_event(request, "auth_error", str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "Internal authentication error"}
        )

    def _log_audit_event(self, request: Request, event_type: str, metadata=None):
        entry = AuditLogEntry(
            timestamp=time.time(),
            event_type=event_type,
            identity=getattr(request.state, "identity", None),
            source_ip=request.client.host,
            user_agent=request.headers.get("User-Agent", ""),
            resource=f"{request.method} {request.url.path}",
            metadata=metadata or {}
        )
        self.kafka.send("audit_logs", entry.dict())

    #endregion

class SecurityPolicy:
    def __init__(self, required_roles=None, required_permissions=None):
        self.required_roles = required_roles or []
        self.required_permissions = required_permissions or []

    def __call__(self, request: Request) -> JWTClaims:
        if not hasattr(request.state, "auth"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        
        claims = request.state.auth
        self._verify_roles(claims)
        self._verify_permissions(claims)
        return claims

    def _verify_roles(self, claims: JWTClaims):
        if self.required_roles and not any(
            role in claims.roles for role in self.required_roles
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges"
            )

    def _verify_permissions(self, claims: JWTClaims):
        if self.required_permissions and not all(
            perm in claims.permissions for perm in self.required_permissions
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing required permissions"
            )

# Example usage in FastAPI
app = FastAPI()
app.add_middleware(
    AuthMiddleware,
    public_routes=["^/healthz$", "^/docs", "^/openapi.json"]
)
