"""
Enterprise Rate Limiter - Distributed Request Throttling with Adaptive Strategies
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp, Receive, Scope, Send

from core.config import RateLimitConfig
from redis.cluster import RedisCluster
from utils.logger import Logger
from utils.metrics import MetricsSystem

#region Data Models

@dataclass(frozen=True)
class RateLimitRule:
    strategy: str  # fixed_window/sliding_window/token_bucket/leaky_bucket
    limit: int     # Max requests
    window: int     # Seconds (for window-based strategies)
    rate: float     # Requests per second (for token/leaky bucket)
    burst: int      # Burst capacity (for token bucket)
    scope: str      # ip/user/api_key/custom
    grouping: List[str]  # Additional grouping keys

@dataclass
class RateLimitState:
    remaining: int
    reset_time: float
    retry_after: float
    limit: int

#endregion

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        redis: RedisCluster = RedisCluster.from_url(os.getenv("REDIS_URL")),
        config: RateLimitConfig = RateLimitConfig(),
        fallback_enabled: bool = True
    ):
        super().__init__(app)
        self.redis = redis
        self.config = config
        self.logger = Logger.get_instance()
        self.metrics = MetricsSystem.get_instance()
        self.fallback_enabled = fallback_enabled
        self.local_cache = {}
        self._strategy_map = {
            "fixed_window": self._fixed_window_handler,
            "sliding_window": self._sliding_window_handler,
            "token_bucket": self._token_bucket_handler,
            "leaky_bucket": self._leaky_bucket_handler
        }
        self._lua_scripts = self._load_lua_scripts()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> JSONResponse:
        client_identity = self._get_client_identity(request)
        path_template = request.scope["route"].path_format if "route" in request.scope else request.url.path

        try:
            # Dynamic rule loading
            rules = await self._get_rules_for_path(path_template)
            if not rules:
                return await call_next(request)

            # Check all applicable rate limits
            violations = []
            for rule in rules:
                state = await self._apply_rate_limit(rule, client_identity, request)
                if state.remaining < 0:
                    violations.append({
                        "limit": state.limit,
                        "retry_after": state.retry_after,
                        "scope": rule.scope,
                        "strategy": rule.strategy
                    })

            if violations:
                return self._rate_limit_response(violations)

            # Process request
            response = await call_next(request)

            # Update token bucket after successful processing
            await self._post_process_token_buckets(rules, client_identity, request)

            return response

        except Exception as e:
            if self.fallback_enabled:
                return await self._fallback_limiter(request, call_next)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit service unavailable"
            )

    #region Core Rate Limit Strategies

    async def _apply_rate_limit(
        self,
        rule: RateLimitRule,
        client_id: str,
        request: Request
    ) -> RateLimitState:
        strategy_fn = self._strategy_map.get(rule.strategy)
        if not strategy_fn:
            raise ValueError(f"Unknown rate limit strategy: {rule.strategy}")

        key = self._build_redis_key(rule, client_id, request)
        return await strategy_fn(key, rule)

    async def _fixed_window_handler(self, key: str, rule: RateLimitRule) -> RateLimitState:
        current_time = time.time()
        window = int(current_time // rule.window) * rule.window

        result = await self.redis.evalsha(
            self._lua_scripts["fixed_window"],
            1,  # Number of keys
            key,
            rule.limit,
            window,
            current_time,
            rule.window
        )

        return RateLimitState(
            remaining=int(result[0]),
            reset_time=float(result[1]),
            retry_after=max(0, float(result[1]) - time.time()),
            limit=rule.limit
        )

    async def _sliding_window_handler(self, key: str, rule: RateLimitRule) -> RateLimitState:
        current_time = time.time()
        precision = min(rule.window, self.config.sliding_window_precision)

        result = await self.redis.evalsha(
            self._lua_scripts["sliding_window"],
            1,
            key,
            rule.limit,
            current_time - rule.window,
            precision,
            current_time
        )

        return RateLimitState(
            remaining=int(result[0]),
            reset_time=current_time + rule.window,
            retry_after=max(0, float(result[1])),
            limit=rule.limit
        )

    #endregion

    #region Token Bucket Implementation

    async def _token_bucket_handler(self, key: str, rule: RateLimitRule) -> RateLimitState:
        current_time = time.time()
        fill_rate = 1.0 / rule.rate
        bucket_capacity = rule.burst if rule.burst else rule.limit

        result = await self.redis.evalsha(
            self._lua_scripts["token_bucket"],
            1,
            key,
            current_time,
            bucket_capacity,
            fill_rate,
            1  # Tokens requested
        )

        remaining = int(float(result[0]))
        reset_time = float(result[1])
        return RateLimitState(
            remaining=remaining,
            reset_time=reset_time,
            retry_after=max(0, (1 - remaining) * fill_rate),
            limit=bucket_capacity
        )

    async def _post_process_token_buckets(
        self,
        rules: List[RateLimitRule],
        client_id: str,
        request: Request
    ):
        for rule in rules:
            if rule.strategy == "token_bucket":
                key = self._build_redis_key(rule, client_id, request)
                await self.redis.hincrby(key, "processed", 1)

    #endregion

    #region Helper Methods

    def _build_redis_key(
        self,
        rule: RateLimitRule,
        client_id: str,
        request: Request
    ) -> str:
        key_parts = [
            "rate_limit",
            rule.scope,
            client_id,
            *[request.headers.get(h) for h in rule.grouping]
        ]
        return hashlib.sha256(":".join(filter(None, key_parts)).encode()).hexdigest()

    def _get_client_identity(self, request: Request) -> str:
        identity_sources = {
            "ip": lambda: request.client.host,
            "user": lambda: request.state.user.id,
            "api_key": lambda: request.headers.get("X-API-Key")
        }
        return ":".join(
            str(source()) 
            for scope, source in identity_sources.items()
            if scope in self.config.identity_scopes
        ) or "anonymous"

    async def _get_rules_for_path(self, path: str) -> List[RateLimitRule]:
        cache_key = f"rl_rules:{path}"
        cached = await self.redis.get(cache_key)
        if cached:
            return [RateLimitRule(**r) for r in json.loads(cached)]

        # Dynamic rule loading from external source
        rules = await self._fetch_dynamic_rules(path)
        await self.redis.setex(cache_key, self.config.rule_cache_ttl, json.dumps(rules))
        return rules

    #endregion

    #region Fallback Mechanism

    async def _fallback_limiter(
        self,
        request: Request,
        call_next: Callable
    ) -> JSONResponse:
        # Local in-memory rate limiter
        client_id = self._get_client_identity(request)
        key = f"local_rl:{client_id}:{request.url.path}"

        current_time = time.time()
        window = self.config.fallback_window

        if key not in self.local_cache:
            self.local_cache[key] = {
                "count": 1,
                "window_start": current_time
            }
        else:
            if current_time - self.local_cache[key]["window_start"] > window:
                self.local_cache[key] = {
                    "count": 1,
                    "window_start": current_time
                }
            else:
                self.local_cache[key]["count"] += 1

        if self.local_cache[key]["count"] > self.config.fallback_limit:
            return self._rate_limit_response([{
                "limit": self.config.fallback_limit,
                "retry_after": window - (current_time - self.local_cache[key]["window_start"]),
                "scope": "local",
                "strategy": "fixed_window"
            }])

        return await call_next(request)

    #endregion

    #region Lua Scripts

    def _load_lua_scripts(self) -> Dict[str, str]:
        scripts = {
            "fixed_window": """
                local key = KEYS[1]
                local limit = tonumber(ARGV[1])
                local window = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])
                local window_sec = tonumber(ARGV[4])
                
                local current = redis.call('GET', key)
                if current and tonumber(current) >= limit then
                    return {tonumber(current), window + window_sec}
                end
                
                redis.call('INCR', key)
                redis.call('EXPIRE', key, window_sec)
                return {limit - 1, window + window_sec}
            """,
            "sliding_window": """
                local key = KEYS[1]
                local limit = tonumber(ARGV[1])
                local oldest = tonumber(ARGV[2])
                local precision = tonumber(ARGV[3])
                local now = tonumber(ARGV[4])
                
                redis.call('ZREMRANGEBYSCORE', key, 0, oldest)
                local count = redis.call('ZCOUNT', key, oldest, now)
                
                if count >= limit then
                    local oldest_entry = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')[2]
                    return {limit - count, oldest_entry + 1}
                end
                
                redis.call('ZADD', key, now, now .. ':' .. math.random())
                redis.call('EXPIRE', key, ARGV[4])
                return {limit - count - 1, 0}
            """,
            "token_bucket": """
                local key = KEYS[1]
                local now = tonumber(ARGV[1])
                local capacity = tonumber(ARGV[2])
                local fill_rate = tonumber(ARGV[3])
                local tokens = tonumber(ARGV[4])
                
                local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
                local last_tokens = tonumber(bucket[1] or capacity)
                local last_update = tonumber(bucket[2] or now)
                
                local delta = math.max(0, now - last_update)
                local filled_tokens = math.min(capacity, last_tokens + delta * fill_rate)
                
                if filled_tokens < tokens then
                    return {filled_tokens, now + (tokens - filled_tokens) / fill_rate}
                end
                
                filled_tokens = filled_tokens - tokens
                redis.call('HMSET', key, 'tokens', filled_tokens, 'last_update', now)
                redis.call('EXPIRE', key, math.ceil(capacity / fill_rate) * 2)
                return {filled_tokens, 0}
            """
        }
        return {name: self.redis.script_load(script) for name, script in scripts.items()}

    #endregion

    #region Response Handling

    def _rate_limit_response(self, violations: List[Dict]) -> JSONResponse:
        self.metrics.increment("rate_limit.violations", tags={
            "path": violations[0].get("scope", "unknown")
        })
        
        headers = {
            "X-RateLimit-Limit": str(violations[0]["limit"]),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(violations[0]["reset_time"])),
            "Retry-After": str(int(violations[0]["retry_after"]))
        }
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "code": "rate_limit_exceeded",
                "message": "Too many requests",
                "violations": violations
            },
            headers=headers
        )

    #endregion

# Configuration Example
config = RateLimitConfig(
    default_rules=[
        RateLimitRule(
            strategy="sliding_window",
            limit=1000,
            window=60,
            scope="ip",
            grouping=["X-API-Version"]
        )
    ],
    sliding_window_precision=1,
    fallback_limit=100,
    fallback_window=60
)

# FastAPI Integration
app.add_middleware(
    RateLimiterMiddleware,
    config=config,
    fallback_enabled=True
)
