#!/usr/bin/env python3
"""
Mixture-of-Agents Tool Module (Command-Code Edition)

Rewritten 2026-06-02: replaced OpenRouter with command-code provider.
Uses floory's actual models: mimo-v2.5-pro, ds-v4-flash, minimax-m3, and
optionally qwen3.7-max.

Architecture (unchanged MoA pattern):
1. Reference models generate diverse initial responses in parallel
2. Aggregator model synthesizes responses into a single high-quality output

Based on: "Mixture-of-Agents Enhances Large Language Model Capabilities"
by Junlin Wang et al. (arXiv:2406.04692v1)

Configuration — modify these constants:
  REFERENCE_MODELS     — list of models for parallel initial responses
  AGGREGATOR_MODEL     — model that synthesizes the final answer
  REFERENCE_TEMPERATURE / AGGREGATOR_TEMPERATURE
  MIN_SUCCESSFUL_REFERENCES — minimum refs needed (otherwise fails)
"""

import json
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI
from agent.auxiliary_client import extract_content_or_reasoning
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION — EDIT THESE
# ══════════════════════════════════════════════════════════════════════════════

# Reference models — diverse perspectives in parallel.
# All via command-code (https://opencode.ai/zen/go/v1).
REFERENCE_MODELS = [
    "mimo-v2.5-pro",        # Moonshot MIMO v2.5 pro — strong multilingual reasoning
    "deepseek-v4-flash",    # DeepSeek V4 Flash — fast, cheap, broad knowledge
    "minimax-m3",           # MiniMax M3 — good creative/analytical mix
    # "qwen3.7-max" is available but ~3-5x more expensive than the others.
    # Uncomment to add as 4th reference when the problem genuinely warrants it.
    # "qwen3.7-max",        # Qwen 3.7 Max — strongest, most expensive
]

# Aggregator — synthesizes all reference responses into final answer.
# DeepSeek V4 Pro is the smartest reasoning model on command-code.
AGGREGATOR_MODEL = "deepseek-v4-pro"

# Temperature: reference models run hotter for diversity, aggregator colder for precision.
REFERENCE_TEMPERATURE = 0.6
AGGREGATOR_TEMPERATURE = 0.3

# Minimum successful reference models needed to proceed with aggregation.
MIN_SUCCESSFUL_REFERENCES = 1

# ══════════════════════════════════════════════════════════════════════════════
# INTERNALS — don't edit unless debugging
# ══════════════════════════════════════════════════════════════════════════════

COMMAND_CODE_BASE_URL = "https://opencode.ai/zen/go/v1"

AGGREGATOR_SYSTEM_PROMPT = (
    "You have been provided with a set of responses from various AI models "
    "to the latest user query. Your task is to synthesize these responses "
    "into a single, high-quality response. It is crucial to critically "
    "evaluate the information provided in these responses, recognizing that "
    "some of it may be biased or incorrect. Your response should not simply "
    "replicate the given answers but should offer a refined, accurate, and "
    "comprehensive reply to the instruction. Ensure your response is "
    "well-structured, coherent, and adheres to the highest standards of "
    "accuracy and reliability.\n\n"
    "Responses from models:"
)

_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    """Lazy-initialize command-code async client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENCODE_GO_API_KEY")
        if not api_key:
            raise ValueError("OPENCODE_GO_API_KEY environment variable not set")
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url=COMMAND_CODE_BASE_URL + "/",
        )
    return _client


def _construct_aggregator_prompt(
    system_prompt: str, responses: List[str]
) -> str:
    """Build aggregator system prompt with enumerated model responses."""
    response_text = "\n".join(
        f"{i+1}. {resp}" for i, resp in enumerate(responses)
    )
    return f"{system_prompt}\n\n{response_text}"


async def _run_reference_model_safe(
    model: str,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_tokens: int = 32000,
    max_retries: int = 3,
) -> tuple:
    """Run one reference model with retries. Returns (model, content, success)."""
    client = _get_client()

    for attempt in range(max_retries):
        try:
            logger.info(
                "Querying %s (attempt %d/%d)", model, attempt + 1, max_retries
            )
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = extract_content_or_reasoning(response)
            if not content:
                logger.warning(
                    "%s returned empty content (attempt %d/%d), retrying",
                    model, attempt + 1, max_retries,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 30))
                    continue
            logger.info("%s responded (%d chars)", model, len(content))
            return model, content, True

        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "limit" in err.lower():
                logger.warning("%s rate limited (attempt %d): %s", model, attempt + 1, err[:120])
            elif "invalid" in err.lower():
                logger.warning("%s invalid request (attempt %d): %s", model, attempt + 1, err[:120])
            else:
                logger.warning("%s error (attempt %d): %s", model, attempt + 1, err[:120])

            if attempt < max_retries - 1:
                delay = min(2 ** (attempt + 1), 30)
                logger.info("Retrying in %ds...", delay)
                await asyncio.sleep(delay)
            else:
                logger.error("%s failed after %d attempts", model, max_retries)
                return model, err, False

    # Shouldn't normally reach here, but satisfy the type checker
    return model, "all retries exhausted", False


async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: int = 32000,
) -> str:
    """Run the aggregator model to synthesize the final response."""
    client = _get_client()

    logger.info("Running aggregator: %s", AGGREGATOR_MODEL)

    response = await client.chat.completions.create(
        model=AGGREGATOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = extract_content_or_reasoning(response)

    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await client.chat.completions.create(
            model=AGGREGATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%d chars)", len(content))
    return content


async def mixture_of_agents_tool(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None,
) -> str:
    """
    Process a complex query through multiple models in parallel, then
    synthesize the best answer.

    Args:
        user_prompt: The problem to solve.
        reference_models: Override default reference models (optional).
        aggregator_model: Override default aggregator model (optional).

    Returns:
        JSON string: {"success": bool, "response": str, "models_used": {...}}
    """
    start_time = datetime.datetime.now()

    ref_models = reference_models or REFERENCE_MODELS
    agg_model = aggregator_model or AGGREGATOR_MODEL

    try:
        logger.info("Starting MoA with %d refs + %s", len(ref_models), agg_model)
        logger.info("Prompt: %s", user_prompt[:200])

        # Layer 1: Parallel reference model calls
        logger.info("Layer 1: %d reference models in parallel", len(ref_models))
        results = await asyncio.gather(*[
            _run_reference_model_safe(m, user_prompt, REFERENCE_TEMPERATURE)
            for m in ref_models
        ])

        successful = []
        failed = []
        for model_name, content, ok in results:
            if ok:
                successful.append(content)
            else:
                failed.append(model_name)

        logger.info(
            "References: %d ok, %d failed%s",
            len(successful), len(failed),
            f" ({', '.join(failed)})" if failed else "",
        )

        if len(successful) < MIN_SUCCESSFUL_REFERENCES:
            raise ValueError(
                f"Insufficient successful references "
                f"({len(successful)}/{len(ref_models)}). "
                f"Need at least {MIN_SUCCESSFUL_REFERENCES}."
            )

        # Layer 2: Aggregation
        logger.info("Layer 2: Aggregating with %s", agg_model)
        system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, successful
        )
        final_response = await _run_aggregator_model(
            system_prompt, user_prompt, AGGREGATOR_TEMPERATURE
        )

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        logger.info("MoA done in %.1fs", elapsed)

        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": ref_models,
                "aggregator_model": agg_model,
                "successful_references": len(successful),
                "failed_references": failed,
            },
            "processing_time": elapsed,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        logger.error("MoA failed: %s", e)

        result = {
            "success": False,
            "response": (
                "MoA processing failed. Please try again or use a single "
                "model for this query."
            ),
            "error": str(e),
            "models_used": {
                "reference_models": ref_models,
                "aggregator_model": agg_model,
            },
            "processing_time": elapsed,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════════════════════

def check_moa_requirements() -> bool:
    """Return True if command-code API key is available."""
    return bool(os.environ.get("OPENCODE_GO_API_KEY"))


def get_moa_configuration() -> Dict[str, Any]:
    """Return current MoA configuration for introspection."""
    return {
        "reference_models": REFERENCE_MODELS,
        "aggregator_model": AGGREGATOR_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "total_reference_models": len(REFERENCE_MODELS),
        "provider": "command-code",
        "base_url": COMMAND_CODE_BASE_URL,
    }


from tools.registry import registry

MOA_SCHEMA = {
    "name": "mixture_of_agents",
    "description": (
        "Route a hard problem through multiple LLMs in parallel, then "
        "synthesize the best answer. Calls 3-4 reference models "
        "(mimo-v2.5-pro, ds-v4-flash, minimax-m3) in parallel, then "
        "aggregates via deepseek-v4-pro. Use sparingly — each call "
        "burns ~$0.15-0.50. Best for: complex math, advanced algorithms, "
        "multi-step reasoning, problems where single models struggle."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": (
                    "The complex query or problem to solve using multiple "
                    "models collaboratively."
                ),
            }
        },
        "required": ["user_prompt"],
    },
}

registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(
        user_prompt=args.get("user_prompt", "")
    ),
    check_fn=check_moa_requirements,
    requires_env=["OPENCODE_GO_API_KEY"],
    is_async=True,
    emoji="🧠",
)
