"LangChain handles this through two 
official prebuilt middlewares working together.

LAYER 1 — ModelRetryMiddleware:
For transient failures — 
network timeout, 500 errors, 
temporary provider issues.

from langchain.agents.middleware import ModelRetryMiddleware

retry = ModelRetryMiddleware(
    max_retries=3,
    backoff_factor=2.0,
    retry_on=(Exception, TimeoutError)
)

Attempt 1: immediate
Attempt 2: wait 2 seconds
Attempt 3: wait 4 seconds
Attempt 4: wait 8 seconds

Transient failures resolve within retries.
User sees no error.

LAYER 2 — ModelFallbackMiddleware:
If all retries fail — provider truly down —
automatically tries next model in chain.

from langchain.agents.middleware import ModelFallbackMiddleware

fallback = ModelFallbackMiddleware(
    'openai:gpt-4o-mini',       # try first on failure
    'anthropic:claude-sonnet-4-6' # then this
)

agent = create_agent(
    model='openai:gpt-4o',   # primary
    middleware=[retry, fallback],
    tools=[]
)

If GPT-4o fails →
automatically tries GPT-4o Mini →
if that fails → tries Claude Sonnet →
if all fail → raises clean error.

KEY DESIGN from official docs:
Middleware runs in SEQUENCE going in.
REVERSE SEQUENCE coming out.

So pipeline is:
Request →
  RetryMiddleware (intercepts, adds retry logic)
  → FallbackMiddleware (intercepts, adds failover)
  → Model call

Response ←
  FallbackMiddleware (inspects result)
  ← RetryMiddleware (inspects result)
  ← Back to agent

Agent never sees provider-level failures.
It sees either a successful response
or a clean structured error
after all middleware has run.

This is the official LangChain 1.0 
production pattern for resilience."