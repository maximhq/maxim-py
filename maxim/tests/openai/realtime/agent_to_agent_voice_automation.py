"""
OpenAI Realtime A<->B Voice Automation for Maxim observability.

This script runs two OpenAI Realtime connections and relays generated audio
between them so no microphone/speaker interaction is required.

Required environment variables:
- OPENAI_API_KEY
- MAXIM_API_KEY
- MAXIM_BASE_URL
- MAXIM_LOG_REPO_ID

Example:
    Main (recommended for pushing logs to the demo log repository):
    uv run --python 3.12 python -m maxim.tests.openai.realtime.agent_to_agent_voice_automation \
        --single-observed-agent \
        --run-scenario-loop

    # Single scenario with defaults (model, voices, scenario, turns):
    uv run --python 3.12 python -m maxim.tests.openai.realtime.agent_to_agent_voice_automation

    # Optional override example:
    uv run --python 3.12 python -m maxim.tests.openai.realtime.agent_to_agent_voice_automation \
        --scenario billing_refund_escalation \
        --turns 14

    # Run default scenario loop (5 scenarios) for more traces:
    uv run --python 3.12 python -m maxim.tests.openai.realtime.agent_to_agent_voice_automation \
        --run-scenario-loop

    # Production-like perspective (log only one observed agent):
    uv run --python 3.12 python -m maxim.tests.openai.realtime.agent_to_agent_voice_automation \
        --single-observed-agent
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from maxim import Maxim
from maxim.logger.components import FileDataAttachment
from maxim.logger.components.types import Entity
from maxim.logger.openai import MaximOpenAIClient
from maxim.logger.utils import pcm16_to_wav_bytes

load_dotenv()

SAMPLE_RATE_HZ = 24_000
BYTES_PER_SAMPLE = 2
CONVERSATION_GAP_MS = 180
STOP_CALL_TOOL = {
    "type": "function",
    "name": "stop_call",
    "description": (
        "End the simulation call only after a realistic multi-step conversation "
        "where key facts are verified, a plan is agreed, and next steps are "
        "acknowledged by both parties."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short reason for ending the call.",
            },
            "outcome": {
                "type": "string",
                "enum": ["resolved", "escalated", "unresolved"],
                "description": "Final outcome of this simulation call.",
            },
        },
        "required": ["reason", "outcome"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ScenarioProfile:
    key: str
    title: str
    description: str
    agent_a_name: str
    agent_a_role: str
    agent_b_name: str
    agent_b_role: str
    agent_a_goal: str
    agent_b_goal: str
    agent_a_playbook: str
    agent_b_playbook: str
    closure_criteria: str
    bootstrap_prompt: str


SCENARIOS: dict[str, ScenarioProfile] = {
    "customer_support_warranty_claim": ScenarioProfile(
        key="customer_support_warranty_claim",
        title="Customer Warranty Claim",
        description=(
            "A customer received a laptop with physical damage and needs a fast, "
            "policy-compliant replacement before urgent travel."
        ),
        agent_a_name="Ava Thompson",
        agent_a_role="Customer",
        agent_b_name="Jordan Rivera",
        agent_b_role="Support Agent",
        agent_a_goal=(
            "Get a clear, realistic replacement plan with timeline and required actions."
        ),
        agent_b_goal=(
            "Collect mandatory verification details, then provide a compliant and "
            "actionable resolution path."
        ),
        agent_a_playbook=(
            "Act like a real customer under time pressure. Do not volunteer all details "
            "up front; answer the current question, then ask one practical follow-up. "
            "Share details progressively: identity/contact, order facts, damage specifics, "
            "and travel deadline. If a detail is unknown, say you need a moment and then "
            "provide a plausible value in a later turn."
        ),
        agent_b_playbook=(
            "Run a staged support flow. Stage 1: verify identity and best contact email. "
            "Stage 2: confirm order ID, purchase date, model/serial, shipping ZIP. "
            "Stage 3: confirm damage location and urgency constraints. Stage 4: propose "
            "a concrete replacement plan with ETA and required customer actions. "
            "Ask one primary question per turn; avoid asking everything at once."
        ),
        closure_criteria=(
            "Only call stop_call after both sides acknowledge all of: final outcome, "
            "shipping method, ETA window, customer next action, and support owner follow-up."
        ),
        bootstrap_prompt=(
            "In English only. You are the customer calling support. "
            "State that order LR-48219 for an OrionBook Pro 14 arrived with a cracked hinge, "
            "you travel in 48 hours, and you need the fastest replacement path. "
            "Do not provide every verification detail in the first turn."
        ),
    ),
    "warranty_replacement_handoff": ScenarioProfile(
        key="warranty_replacement_handoff",
        title="Warranty Replacement Handoff",
        description=(
            "A premium laptop arrived with a cracked hinge two days before the "
            "customer's business trip. Team must choose the fastest feasible "
            "replacement path with clear commitments."
        ),
        agent_a_name="Nina Patel",
        agent_a_role="Tier-1 Support Lead",
        agent_b_name="Marcus Lee",
        agent_b_role="Logistics Specialist",
        agent_a_goal=(
            "Gather the core issue and keep customer-impact communication precise."
        ),
        agent_b_goal=(
            "Confirm inventory and shipping options, then propose a realistic plan."
        ),
        agent_a_playbook=(
            "Lead a structured handoff. Start with customer impact and known facts, then "
            "request missing operational details one topic at a time: stock location, "
            "cutoff time, carrier SLA, and fallback option. Confirm ownership clearly."
        ),
        agent_b_playbook=(
            "Respond like logistics operations. Validate one constraint per turn: "
            "inventory location, pickup cutoff, overnight feasibility, and fallback route. "
            "Give realistic ETAs and assumptions, not generic promises."
        ),
        closure_criteria=(
            "Call stop_call only after both sides agree: primary plan, fallback plan, "
            "ticket owner, escalation condition, and customer-facing ETA."
        ),
        bootstrap_prompt=(
            "In English only. Start the handoff by sharing order ID LR-48219, "
            "device model OrionBook Pro 14, and that the hinge cracked on arrival. "
            "Ask for fastest replacement options before the customer travels in 48 hours. "
            "Start with only high-level context and request details step by step."
        ),
    ),
    "billing_refund_escalation": ScenarioProfile(
        key="billing_refund_escalation",
        title="Billing Refund Escalation",
        description=(
            "A customer was double-charged for an annual subscription and needs a "
            "clear refund timeline plus temporary access continuity."
        ),
        agent_a_name="Ava Thompson",
        agent_a_role="Customer",
        agent_b_name="Jordan Rivera",
        agent_b_role="Support Agent",
        agent_a_goal="Get a clear refund plan and ensure service continuity.",
        agent_b_goal="Validate charge discrepancy and provide executable refund steps.",
        agent_a_playbook=(
            "Be a realistic customer who was double-charged and is worried about access. "
            "Share transaction details gradually as requested, and ask clarifying questions "
            "about timeline, interim access, and confirmation email."
        ),
        agent_b_playbook=(
            "Use a phased billing workflow. Collect verification in sequence: account email, "
            "invoice ID, charge amount/date, card last4, and whether charges are pending or posted. "
            "Then provide refund path, expected bank posting window, and interim access policy. "
            "Ask one main question at a time."
        ),
        closure_criteria=(
            "Call stop_call only after customer confirms understanding of refund steps, "
            "expected refund timeline, interim access status, and next update channel."
        ),
        bootstrap_prompt=(
            "In English only. Start by stating invoice INV-7734 was charged twice "
            "for $249, and ask for refund timing plus whether access remains active. "
            "Do not include card last4 or every account detail immediately."
        ),
    ),
    "customer_support_missing_delivery": ScenarioProfile(
        key="customer_support_missing_delivery",
        title="Missing Delivery Follow-up",
        description=(
            "A customer's order was marked delivered but the package is missing. "
            "Support must verify details and provide a clear next-step path."
        ),
        agent_a_name="Ava Thompson",
        agent_a_role="Customer",
        agent_b_name="Jordan Rivera",
        agent_b_role="Support Agent",
        agent_a_goal=(
            "Report missing package clearly and get a concrete resolution path quickly."
        ),
        agent_b_goal=(
            "Collect required delivery-verification details and provide a policy-compliant next action."
        ),
        agent_a_playbook=(
            "Be a realistic customer who already did some checks but still needs help. "
            "Provide details progressively: delivery timestamp, ZIP, checks completed, and "
            "any delivery note details. Ask what happens next and when you will get updates."
        ),
        agent_b_playbook=(
            "Handle this as a carrier-claim workflow. Verify details in sequence: order ID, "
            "delivery timestamp, ZIP, safe-drop/neighbor/mailroom checks, and delivery note wording. "
            "Then choose one path (investigation, replacement, or claim) with SLA, owner, and customer action."
        ),
        closure_criteria=(
            "Call stop_call only after both parties confirm investigation or replacement path, "
            "owner of next step, ETA for update, and what the customer should do meanwhile."
        ),
        bootstrap_prompt=(
            "In English only. You are the customer. Say order PK-90311 was marked delivered "
            "yesterday at 6:12 PM but no package is found at your door or mailroom. "
            "Share only the basics first and ask support what they need to verify."
        ),
    ),
    "account_security_takeover_recovery": ScenarioProfile(
        key="account_security_takeover_recovery",
        title="Account Takeover Recovery",
        description=(
            "A customer reports suspicious account activity and is locked out after reset attempts. "
            "Support must secure the account, verify identity, and restore safe access."
        ),
        agent_a_name="Ava Thompson",
        agent_a_role="Customer",
        agent_b_name="Jordan Rivera",
        agent_b_role="Security Support Agent",
        agent_a_goal=(
            "Regain secure account access quickly and understand immediate safety steps."
        ),
        agent_b_goal=(
            "Run identity verification, lock down risk vectors, and provide a safe recovery path."
        ),
        agent_a_playbook=(
            "Be concerned but cooperative. Share account details progressively as requested, "
            "including email, last successful login, recent suspicious alerts, and available verification methods. "
            "Ask practical questions about time-to-recovery and what to do immediately."
        ),
        agent_b_playbook=(
            "Use a staged security flow. Stage 1: verify identity with required checkpoints. "
            "Stage 2: confirm suspicious activity details (time, location, device). "
            "Stage 3: apply containment actions (session revocation, password reset, 2FA reset/rebind). "
            "Stage 4: confirm secure re-entry steps and follow-up monitoring guidance. "
            "Ask one main question per turn."
        ),
        closure_criteria=(
            "Call stop_call only after both parties confirm: containment actions completed, "
            "safe access recovery path, immediate next customer action, and security follow-up timeline."
        ),
        bootstrap_prompt=(
            "In English only. You are the customer, speaking in first person. "
            "Start by reporting that your account tied to ava.t@mail.net showed unfamiliar login alerts "
            "from another city and you are now locked out after reset attempts. "
            "Ask for urgent help securing and recovering access. "
            "Do not sound like support, do not apologize to yourself, and do not ask verification questions."
        ),
    ),
}

DEFAULT_SCENARIO_LOOP = [
    "customer_support_warranty_claim",
    "billing_refund_escalation",
    "customer_support_missing_delivery",
    "warranty_replacement_handoff",
    "account_security_takeover_recovery",
]


@dataclass(frozen=True)
class AutomationConfig:
    turns: int = 18
    model: str = "gpt-realtime"
    scenario: str = "customer_support_warranty_claim"
    voice_a: str = "alloy"
    voice_b: str = "marin"
    bootstrap_instructions: str | None = None
    chunk_ms: int = 40
    turn_timeout_sec: float = 45.0
    session_name: str = "OpenAI Realtime A-B Voice"
    debug_events: bool = False
    max_failures: int = 2
    observability_mode: str = "dual"


@dataclass
class TurnResult:
    turn_index: int
    speaker: str
    speaker_label: str
    response_id: str | None
    response_status: str
    audio_bytes: int
    transcription_events: int
    assistant_transcript: str | None
    tool_calls: list[str]
    stop_call_call_id: str | None
    stop_call_reason: str | None
    stop_call_outcome: str | None
    relayed: bool
    text_only: bool
    audio_data: bytes = field(default_factory=bytes, repr=False)
    errors: list[str] = field(default_factory=list)


@dataclass
class AutomationRunSummary:
    session_id: str
    session_name: str
    observability_mode: str
    turns_requested: int
    turns_completed: int
    stop_reason: str
    per_agent_response_statuses: dict[str, list[str]]
    per_agent_response_done_count: dict[str, int]
    text_only_turns: int
    error_count: int
    elapsed_seconds: float
    turn_results: list[TurnResult]


@dataclass
class AgentRuntime:
    name: str
    display_name: str
    role: str
    connection: Any
    voice: str
    event_queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)


def _b64(blob: bytes) -> str:
    return base64.b64encode(blob).decode("utf-8")


def _extract_response_status(response: Any) -> str:
    status = getattr(response, "status", None)
    if isinstance(status, str):
        return status
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        dumped_status = dumped.get("status")
        if isinstance(dumped_status, str):
            return dumped_status
    if isinstance(response, dict):
        dict_status = response.get("status")
        if isinstance(dict_status, str):
            return dict_status
    return "unknown"


def _extract_response_id(response: Any) -> str | None:
    response_id = getattr(response, "id", None)
    if isinstance(response_id, str):
        return response_id
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        dumped_id = dumped.get("id")
        if isinstance(dumped_id, str):
            return dumped_id
    if isinstance(response, dict):
        dict_id = response.get("id")
        if isinstance(dict_id, str):
            return dict_id
    return None


def _extract_error_message(error_obj: Any) -> str:
    if isinstance(error_obj, dict):
        message = error_obj.get("message")
        if isinstance(message, str):
            return message
        return str(error_obj)
    message = getattr(error_obj, "message", None)
    if isinstance(message, str):
        return message
    return str(error_obj)


def _append_unique_text(parts: list[str], candidate: str | None) -> None:
    if not isinstance(candidate, str):
        return
    cleaned = candidate.strip()
    if not cleaned:
        return
    if not parts or parts[-1] != cleaned:
        parts.append(cleaned)


def _extract_response_transcript(response: Any) -> str | None:
    payload: dict[str, Any] | None = None
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    if not payload:
        return None

    output = payload.get("output")
    if not isinstance(output, list):
        return None

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        contents = item.get("content")
        if not isinstance(contents, list):
            continue
        for content in contents:
            if not isinstance(content, dict):
                continue
            content_type = content.get("type")
            if content_type not in {"output_audio", "output_text"}:
                continue
            _append_unique_text(parts, content.get("transcript"))
            _append_unique_text(parts, content.get("text"))
    if not parts:
        return None
    return " ".join(parts)


def _get_scenario_profile(scenario_key: str) -> ScenarioProfile:
    if scenario_key not in SCENARIOS:
        allowed = ", ".join(sorted(SCENARIOS.keys()))
        raise ValueError(f"Unknown scenario '{scenario_key}'. Allowed: {allowed}")
    return SCENARIOS[scenario_key]


def _build_agent_instruction(
    *,
    scenario: ScenarioProfile,
    is_agent_a: bool,
) -> str:
    me_name = scenario.agent_a_name if is_agent_a else scenario.agent_b_name
    me_role = scenario.agent_a_role if is_agent_a else scenario.agent_b_role
    me_goal = scenario.agent_a_goal if is_agent_a else scenario.agent_b_goal
    peer_name = scenario.agent_b_name if is_agent_a else scenario.agent_a_name
    peer_role = scenario.agent_b_role if is_agent_a else scenario.agent_a_role
    me_playbook = scenario.agent_a_playbook if is_agent_a else scenario.agent_b_playbook
    me_role_lower = me_role.lower()

    role_integrity = (
        f"Role lock: You must always speak as {me_name} ({me_role}) and never as "
        f"{peer_name} ({peer_role}). Never switch roles."
    )
    if "customer" in me_role_lower:
        role_integrity += (
            " As customer: speak in first person about your own issue. "
            "Never use support-agent phrasing like 'I understand your issue', "
            "'let's get this sorted for you', or identity-verification questioning."
        )
    if "support" in me_role_lower or "specialist" in me_role_lower:
        role_integrity += (
            " As support/operations: do not role-play as the customer; "
            "ask clarifying questions and provide concrete policy/ops steps."
        )

    return (
        f"You are {me_name}, {me_role}. "
        f"You are speaking with {peer_name}, {peer_role}, to handle this scenario: "
        f"{scenario.description} "
        "Speak only in English. Keep each turn concise (one or two short sentences). "
        f"Your objective: {me_goal} "
        f"Role instructions: {me_playbook} "
        f"{role_integrity} "
        "Always be realistic and operationally specific. "
        "Use progressive dialogue: ask or answer one primary point per turn instead of bundling everything. "
        "Do not rush to closure; move through discovery, verification, plan proposal, and final confirmation. "
        "Target a realistic call length of around 6 to 10 turns, and avoid ending in under 5 turns unless no further progress is possible. "
        "Use the stop_call tool only when the conversation should end because the case "
        "is resolved, escalated, or not solvable in this call. "
        f"Closure criteria: {scenario.closure_criteria} "
        "Before calling stop_call, ensure both sides have acknowledged the final outcome and next-step ownership. "
        "When calling stop_call, provide a short reason and valid outcome."
    )


def _extract_response_function_calls(response: Any) -> list[dict[str, str]]:
    payload: dict[str, Any] | None = None
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    if not payload:
        return []

    output = payload.get("output")
    if not isinstance(output, list):
        return []

    function_calls: list[dict[str, str]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str):
            continue
        arguments = item.get("arguments")
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        elif isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = ""
        function_calls.append(
            {"call_id": call_id, "name": name, "arguments": arguments_str}
        )
    return function_calls


def _extract_stop_call(
    function_calls: list[dict[str, str]],
) -> tuple[str, str | None, str | None] | None:
    for call in function_calls:
        if call.get("name") != "stop_call":
            continue
        call_id = call.get("call_id")
        if not isinstance(call_id, str):
            continue

        reason: str | None = None
        outcome: str | None = None
        args_raw = call.get("arguments")
        if isinstance(args_raw, str) and args_raw.strip():
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    reason_raw = parsed.get("reason")
                    outcome_raw = parsed.get("outcome")
                    if isinstance(reason_raw, str):
                        reason = reason_raw.strip() or None
                    if isinstance(outcome_raw, str):
                        outcome = outcome_raw.strip() or None
            except json.JSONDecodeError:
                reason = args_raw.strip() or None

        return (call_id, reason, outcome)
    return None


def _iter_chunks(blob: bytes, chunk_size: int):
    for i in range(0, len(blob), chunk_size):
        yield blob[i : i + chunk_size]


def _validate_config(config: AutomationConfig) -> None:
    if config.turns <= 0:
        raise ValueError("--turns must be > 0")
    if config.chunk_ms <= 0:
        raise ValueError("--chunk-ms must be > 0")
    if config.turn_timeout_sec <= 0:
        raise ValueError("--turn-timeout-sec must be > 0")
    if config.max_failures <= 0:
        raise ValueError("--max-failures must be > 0")
    if config.observability_mode not in {"dual", "single-agent-a", "single-agent-b"}:
        raise ValueError(
            "--observability-mode must be one of: dual, single-agent-a, single-agent-b"
        )
    _get_scenario_profile(config.scenario)


def _observed_agents_for_mode(mode: str) -> tuple[bool, bool]:
    if mode == "dual":
        return (True, True)
    if mode == "single-agent-a":
        return (True, False)
    if mode == "single-agent-b":
        return (False, True)
    raise ValueError(
        f"Unknown observability mode '{mode}'. "
        "Allowed: dual, single-agent-a, single-agent-b"
    )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


async def _configure_connection(
    connection: Any,
    *,
    model: str,
    voice: str,
    instructions: str,
    tools: list[dict[str, Any]],
) -> None:
    await connection.session.update(
        session={
            "model": model,
            "type": "realtime",
            "output_modalities": ["audio"],
            "tracing": "auto",
            "instructions": instructions,
            "tool_choice": "auto",
            "tools": tools,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                    "turn_detection": None,
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    "voice": voice,
                },
            },
        }
    )


async def _connection_listener(agent: AgentRuntime, debug_events: bool) -> None:
    try:
        async for event in agent.connection:
            event_type = getattr(event, "type", "")

            if debug_events:
                print(f"[{agent.name}] event: {event_type}")

            if event_type == "response.output_audio.delta":
                await agent.event_queue.put(
                    {
                        "type": event_type,
                        "delta": getattr(event, "delta", None),
                    }
                )
            elif event_type == "response.done":
                await agent.event_queue.put(
                    {
                        "type": event_type,
                        "response": getattr(event, "response", None),
                    }
                )
            elif event_type == "response.output_audio_transcript.done":
                await agent.event_queue.put(
                    {
                        "type": event_type,
                        "transcript": getattr(event, "transcript", None),
                    }
                )
            elif event_type == "response.output_text.done":
                await agent.event_queue.put(
                    {
                        "type": event_type,
                        "text": getattr(event, "text", None),
                    }
                )
            elif event_type == "session.updated":
                await agent.event_queue.put({"type": event_type})
            elif event_type == "conversation.item.input_audio_transcription.completed":
                await agent.event_queue.put(
                    {
                        "type": event_type,
                        "item_id": getattr(event, "item_id", None),
                        "transcript": getattr(event, "transcript", None),
                    }
                )
            elif event_type in ("realtime.error", "error"):
                await agent.event_queue.put(
                    {
                        "type": "error",
                        "message": _extract_error_message(getattr(event, "error", None)),
                    }
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        await agent.event_queue.put(
            {
                "type": "error",
                "message": f"listener_failure:{exc}",
            }
        )


async def _await_session_updated(agent: AgentRuntime, timeout_sec: float = 12.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{agent.name} did not emit session.updated in time")
        queued_event = await asyncio.wait_for(agent.event_queue.get(), timeout=remaining)
        event_type = queued_event.get("type")
        if event_type == "session.updated":
            return
        if event_type == "error":
            message = str(queued_event.get("message", "unknown_error"))
            raise RuntimeError(f"{agent.name} session.update failed: {message}")


async def _drain_post_done_events(
    event_queue: asyncio.Queue[dict[str, Any]],
    duration_sec: float,
) -> list[dict[str, Any]]:
    drained: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration_sec
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        timeout = min(0.05, remaining)
        try:
            drained.append(await asyncio.wait_for(event_queue.get(), timeout=timeout))
        except asyncio.TimeoutError:
            break
    return drained


async def _collect_turn_result(
    agent: AgentRuntime,
    turn_index: int,
    timeout_sec: float,
) -> TurnResult:
    deadline = time.monotonic() + timeout_sec
    audio_chunks: list[bytes] = []
    transcription_events = 0
    errors: list[str] = []
    response_id: str | None = None
    response_status = "timeout"
    assistant_transcript_parts: list[str] = []
    response_obj: Any = None
    function_calls: list[dict[str, str]] = []

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            errors.append("turn_timeout")
            break

        queued_event = await asyncio.wait_for(agent.event_queue.get(), timeout=remaining)
        event_type = queued_event["type"]

        if event_type == "response.output_audio.delta":
            delta = queued_event.get("delta")
            if isinstance(delta, str):
                try:
                    audio_chunks.append(base64.b64decode(delta))
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append(f"audio_decode_error:{exc}")
        elif event_type == "response.output_audio_transcript.done":
            _append_unique_text(
                assistant_transcript_parts,
                queued_event.get("transcript"),
            )
        elif event_type == "response.output_text.done":
            _append_unique_text(
                assistant_transcript_parts,
                queued_event.get("text"),
            )
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcription_events += 1
        elif event_type == "error":
            errors.append(str(queued_event.get("message")))
        elif event_type == "response.done":
            response_obj = queued_event.get("response")
            response_status = _extract_response_status(response_obj)
            response_id = _extract_response_id(response_obj)
            function_calls = _extract_response_function_calls(response_obj)
            _append_unique_text(
                assistant_transcript_parts,
                _extract_response_transcript(response_obj),
            )
            break
        elif event_type == "session.updated":
            # This can still appear late if queue wasn't fully drained between setup and turns.
            continue

    for queued_event in await _drain_post_done_events(agent.event_queue, duration_sec=0.25):
        event_type = queued_event.get("type")
        if event_type == "response.output_audio.delta":
            delta = queued_event.get("delta")
            if isinstance(delta, str):
                try:
                    audio_chunks.append(base64.b64decode(delta))
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append(f"audio_decode_error:{exc}")
        elif event_type == "response.output_audio_transcript.done":
            _append_unique_text(
                assistant_transcript_parts,
                queued_event.get("transcript"),
            )
        elif event_type == "response.output_text.done":
            _append_unique_text(
                assistant_transcript_parts,
                queued_event.get("text"),
            )
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcription_events += 1
        elif event_type == "error":
            errors.append(str(queued_event.get("message")))
        elif event_type == "response.done" and response_obj is None:
            response_obj = queued_event.get("response")
            response_status = _extract_response_status(response_obj)
            response_id = _extract_response_id(response_obj)
            function_calls = _extract_response_function_calls(response_obj)
            _append_unique_text(
                assistant_transcript_parts,
                _extract_response_transcript(response_obj),
            )

    stop_call = _extract_stop_call(function_calls)
    stop_call_id = stop_call[0] if stop_call else None
    stop_call_reason = stop_call[1] if stop_call else None
    stop_call_outcome = stop_call[2] if stop_call else None

    full_audio = b"".join(audio_chunks)
    return TurnResult(
        turn_index=turn_index,
        speaker=agent.name,
        speaker_label=agent.display_name,
        response_id=response_id,
        response_status=response_status,
        audio_bytes=len(full_audio),
        transcription_events=transcription_events,
        assistant_transcript=" ".join(assistant_transcript_parts)
        if assistant_transcript_parts
        else None,
        tool_calls=[call["name"] for call in function_calls if call.get("name")],
        stop_call_call_id=stop_call_id,
        stop_call_reason=stop_call_reason,
        stop_call_outcome=stop_call_outcome,
        relayed=False,
        text_only=len(full_audio) == 0,
        audio_data=full_audio,
        errors=errors,
    )


async def _relay_audio_and_trigger_response(
    target: AgentRuntime,
    audio_bytes: bytes,
    chunk_size_bytes: int,
) -> None:
    for chunk in _iter_chunks(audio_bytes, chunk_size_bytes):
        await target.connection.input_audio_buffer.append(audio=_b64(chunk))
    await target.connection.input_audio_buffer.commit()
    await target.connection.response.create(
        response={
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "voice": target.voice,
                }
            },
        }
    )


async def _submit_stop_call_output(
    agent: AgentRuntime,
    call_id: str,
    reason: str | None,
    outcome: str | None,
) -> None:
    payload = {
        "status": "call_ended",
        "ended_by": agent.display_name,
        "outcome": outcome or "resolved",
        "reason": reason or "Conversation reached a natural conclusion.",
    }
    await agent.connection.conversation.item.create(
        item={
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(payload),
        }
    )


async def _cancel_tasks(tasks: list[asyncio.Task[Any]]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _make_runtime_headers(
    session_id: str,
    session_name: str,
    generation_name: str,
    scenario_key: str,
) -> dict[str, str]:
    session_tags = json.dumps(
        {
            "scenario": scenario_key,
        }
    )
    return {
        "maxim-session-id": session_id,
        "maxim-session-name": session_name,
        "maxim-generation-name": generation_name,
        "maxim-session-tags": session_tags,
    }


def _build_conversation_audio(turns: list[TurnResult]) -> bytes:
    gap_bytes = (
        int(SAMPLE_RATE_HZ * CONVERSATION_GAP_MS / 1000) * BYTES_PER_SAMPLE
    )
    gap = b"\x00" * gap_bytes
    parts: list[bytes] = []
    for turn in turns:
        if turn.audio_data:
            parts.append(turn.audio_data)
            parts.append(gap)
    if not parts:
        return b""
    # remove trailing silence
    if parts[-1] == gap:
        parts.pop()
    return b"".join(parts)


def _build_turns_only_transcript(turns: list[TurnResult]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"{turn.speaker_label}: {turn.assistant_transcript or ''}")
    return "\n".join(lines)


def _build_shared_logger_and_clients() -> tuple[Any, Any, Any]:
    openai_api_key = _require_env("OPENAI_API_KEY")
    maxim_api_key = _require_env("MAXIM_API_KEY")
    maxim_base_url = _require_env("MAXIM_BASE_URL")
    maxim_log_repo_id = _require_env("MAXIM_LOG_REPO_ID")

    logger = Maxim(
        {
            "api_key": maxim_api_key,
            "base_url": maxim_base_url,
        }
    ).logger({"id": maxim_log_repo_id, "auto_flush": False})
    client_aio = MaximOpenAIClient(OpenAI(api_key=openai_api_key), logger=logger).aio
    plain_client_aio = AsyncOpenAI(api_key=openai_api_key)
    return logger, client_aio, plain_client_aio


def _flush_logger_sync_best_effort(logger: Any) -> None:
    """
    Best-effort synchronous flush.
    Used at scenario end so session_end is emitted after attachment uploads.
    """
    writer = getattr(logger, "writer", None)
    if writer is None:
        try:
            logger.flush()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"warning: failed to flush logger: {exc}")
        return
    try:
        if hasattr(writer, "flush_upload_attachment_logs"):
            writer.flush_upload_attachment_logs(is_sync=True)
        if hasattr(writer, "flush_commit_logs"):
            writer.flush_commit_logs(is_sync=True)
        elif hasattr(writer, "flush"):
            try:
                writer.flush(is_sync=True)
            except TypeError:
                writer.flush()
        else:
            logger.flush()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"warning: failed to sync flush logger: {exc}")


def _install_trace_metadata_workarounds(
    logger: Any,
    session_name: str,
) -> Callable[[], None]:
    """
    Script-only workarounds (without changing SDK core wrappers):
    1) Tag each newly created trace with session_name.
    2) Rename trace title to "Realtime - Turn N" for clearer turn navigation.
    """
    writer = getattr(logger, "writer", None)
    if writer is None or not hasattr(writer, "commit"):
        return lambda: None

    original_commit = writer.commit
    state = {"inside": False, "trace_turn_index": 0}

    def patched_commit(log):  # type: ignore[no-untyped-def]
        if state["inside"]:
            original_commit(log)
            return
        try:
            is_trace_create = (
                getattr(log, "entity", None) == Entity.TRACE
                and getattr(log, "action", None) == "create"
            )
            if is_trace_create:
                trace_data = getattr(log, "data", None)
                if isinstance(trace_data, dict):
                    state["trace_turn_index"] += 1
                    trace_data["name"] = (
                        f"Realtime - Turn {state['trace_turn_index']}"
                    )

            original_commit(log)

            if (
                is_trace_create
            ):
                trace_id = getattr(log, "entity_id", None)
                if isinstance(trace_id, str):
                    state["inside"] = True
                    try:
                        logger.trace_add_tag(trace_id, "session_name", session_name)
                    except Exception:
                        pass
        finally:
            state["inside"] = False

    writer.commit = patched_commit

    def restore() -> None:
        writer.commit = original_commit

    return restore


async def run_duplex_voice_automation(
    config: AutomationConfig,
    *,
    logger: Any | None = None,
    client_aio: Any | None = None,
    plain_client_aio: Any | None = None,
    flush_logger: bool = True,
) -> AutomationRunSummary:
    _validate_config(config)
    scenario = _get_scenario_profile(config.scenario)
    bootstrap_instructions = config.bootstrap_instructions or scenario.bootstrap_prompt
    effective_session_name = (
        f"OpenAI Realtime - {scenario.title}"
        if config.session_name == AutomationConfig.session_name
        else config.session_name
    )

    if (logger is None) != (client_aio is None):
        raise ValueError("Pass both logger and client_aio, or neither.")
    owns_logger = False
    if logger is None and client_aio is None and plain_client_aio is None:
        logger, client_aio, plain_client_aio = _build_shared_logger_and_clients()
        owns_logger = True
    elif logger is not None and client_aio is not None and plain_client_aio is None:
        plain_client_aio = AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
    elif (
        logger is None
        or client_aio is None
        or plain_client_aio is None
    ):
        raise ValueError(
            "Pass logger, client_aio, and plain_client_aio together, or pass none."
        )

    session_id = str(uuid4())
    chunk_size = int(SAMPLE_RATE_HZ * config.chunk_ms / 1000) * BYTES_PER_SAMPLE
    start_time = time.monotonic()

    observe_agent_a, observe_agent_b = _observed_agents_for_mode(
        config.observability_mode
    )
    client_for_agent_a = client_aio if observe_agent_a else plain_client_aio
    client_for_agent_b = client_aio if observe_agent_b else plain_client_aio

    statuses: dict[str, list[str]] = {"agent_a": [], "agent_b": []}
    response_done_count: dict[str, int] = {"agent_a": 0, "agent_b": 0}
    turn_results: list[TurnResult] = []
    error_count = 0
    text_only_turns = 0
    consecutive_failures = 0
    stop_reason = "max_turns_reached"
    restore_trace_metadata_hook = _install_trace_metadata_workarounds(
        logger, effective_session_name
    )

    headers_a = (
        _make_runtime_headers(
            session_id=session_id,
            session_name=effective_session_name,
            generation_name=f"{scenario.agent_a_name} Voice Turn",
            scenario_key=scenario.key,
        )
        if observe_agent_a
        else None
    )
    headers_b = (
        _make_runtime_headers(
            session_id=session_id,
            session_name=effective_session_name,
            generation_name=f"{scenario.agent_b_name} Voice Turn",
            scenario_key=scenario.key,
        )
        if observe_agent_b
        else None
    )

    listener_tasks: list[asyncio.Task[Any]] = []
    session_ready_for_end = False

    try:
        conn_a_kwargs: dict[str, Any] = {"model": config.model}
        conn_b_kwargs: dict[str, Any] = {"model": config.model}
        if headers_a:
            conn_a_kwargs["extra_headers"] = headers_a
        if headers_b:
            conn_b_kwargs["extra_headers"] = headers_b

        async with client_for_agent_a.realtime.connect(
            **conn_a_kwargs
        ) as conn_a, client_for_agent_b.realtime.connect(
            **conn_b_kwargs
        ) as conn_b:
            agent_a = AgentRuntime(
                name="agent_a",
                display_name=scenario.agent_a_name,
                role=scenario.agent_a_role,
                connection=conn_a,
                voice=config.voice_a,
            )
            agent_b = AgentRuntime(
                name="agent_b",
                display_name=scenario.agent_b_name,
                role=scenario.agent_b_role,
                connection=conn_b,
                voice=config.voice_b,
            )

            listener_tasks.append(
                asyncio.create_task(
                    _connection_listener(agent_a, debug_events=config.debug_events)
                )
            )
            listener_tasks.append(
                asyncio.create_task(
                    _connection_listener(agent_b, debug_events=config.debug_events)
                )
            )

            await _configure_connection(
                conn_a,
                model=config.model,
                voice=config.voice_a,
                instructions=_build_agent_instruction(
                    scenario=scenario, is_agent_a=True
                ),
                tools=[STOP_CALL_TOOL],
            )
            await _configure_connection(
                conn_b,
                model=config.model,
                voice=config.voice_b,
                instructions=_build_agent_instruction(
                    scenario=scenario, is_agent_a=False
                ),
                tools=[STOP_CALL_TOOL],
            )
            await _await_session_updated(agent_a)
            await _await_session_updated(agent_b)
            session_ready_for_end = True

            await conn_a.response.create(
                response={
                    "instructions": bootstrap_instructions,
                    "output_modalities": ["audio"],
                    "audio": {
                        "output": {
                            "voice": config.voice_a,
                        }
                    },
                }
            )

            current_speaker = agent_a

            for turn_index in range(1, config.turns + 1):
                turn_result = await _collect_turn_result(
                    current_speaker,
                    turn_index=turn_index,
                    timeout_sec=config.turn_timeout_sec,
                )
                statuses[current_speaker.name].append(turn_result.response_status)
                if turn_result.response_status != "timeout":
                    response_done_count[current_speaker.name] += 1
                error_count += len(turn_result.errors)

                if turn_result.response_status != "completed":
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0

                if turn_result.response_status == "timeout":
                    stop_reason = "timeout_threshold_reached"
                    turn_results.append(turn_result)
                    break

                if turn_result.stop_call_call_id:
                    try:
                        await _submit_stop_call_output(
                            current_speaker,
                            turn_result.stop_call_call_id,
                            turn_result.stop_call_reason,
                            turn_result.stop_call_outcome,
                        )
                        await asyncio.sleep(0.2)
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        turn_result.errors.append(f"stop_call_output_error:{exc}")
                        error_count += 1

                    stop_reason = f"stop_call_{turn_result.stop_call_outcome or 'requested'}"
                    turn_results.append(turn_result)
                    break

                if turn_result.text_only:
                    text_only_turns += 1
                    stop_reason = "text_only_turn_no_audio_relay"
                    turn_results.append(turn_result)
                    break

                if consecutive_failures >= config.max_failures:
                    stop_reason = "repeated_failures"
                    turn_results.append(turn_result)
                    break

                relay_target = agent_b if current_speaker is agent_a else agent_a
                try:
                    await _relay_audio_and_trigger_response(
                        relay_target,
                        turn_result.audio_data,
                        chunk_size_bytes=chunk_size,
                    )
                    turn_result.relayed = True
                    current_speaker = relay_target
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    turn_result.errors.append(f"relay_error:{exc}")
                    error_count += 1
                    consecutive_failures += 1
                    stop_reason = (
                        "repeated_failures"
                        if consecutive_failures >= config.max_failures
                        else "relay_error"
                    )
                    turn_results.append(turn_result)
                    break

                turn_results.append(turn_result)
            else:
                stop_reason = "max_turns_reached"
    finally:
        restore_trace_metadata_hook()
        if turn_results:
            try:
                transcript = _build_turns_only_transcript(turn_results)
                logger.session_add_attachment(
                    session_id,
                    FileDataAttachment(
                        data=transcript.encode("utf-8"),
                        name="Transcript",
                        mime_type="text/plain",
                        tags={
                            "type": "conversation-transcript",
                        },
                        timestamp=int(time.time()),
                    ),
                )
                conversation_audio_pcm = _build_conversation_audio(turn_results)
                if conversation_audio_pcm:
                    logger.session_add_attachment(
                        session_id,
                        FileDataAttachment(
                            data=pcm16_to_wav_bytes(conversation_audio_pcm),
                            name="Full Conversation Audio",
                            mime_type="audio/wav",
                            tags={
                                "type": "conversation-audio",
                            },
                            timestamp=int(time.time()),
                        ),
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error_count += 1
                print(
                    f"warning: failed to attach session transcript/conversation audio: {exc}"
                )
        await _cancel_tasks(listener_tasks)
        if session_ready_for_end:
            if flush_logger:
                _flush_logger_sync_best_effort(logger)
            try:
                logger.session_end(session_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error_count += 1
                print(f"warning: failed to end session {session_id}: {exc}")
            if flush_logger:
                _flush_logger_sync_best_effort(logger)
        elif flush_logger:
            logger.flush()
        # Ensure all pending logs/attachments are pushed before process exit for
        # locally-created logger instances.
        if owns_logger:
            logger.cleanup(is_sync=True)

    elapsed_seconds = time.monotonic() - start_time
    return AutomationRunSummary(
        session_id=session_id,
        session_name=effective_session_name,
        observability_mode=config.observability_mode,
        turns_requested=config.turns,
        turns_completed=len(turn_results),
        stop_reason=stop_reason,
        per_agent_response_statuses=statuses,
        per_agent_response_done_count=response_done_count,
        text_only_turns=text_only_turns,
        error_count=error_count,
        elapsed_seconds=elapsed_seconds,
        turn_results=turn_results,
    )


def _print_summary(summary: AutomationRunSummary) -> None:
    print("\n=== OpenAI Realtime A<->B Automation Summary ===")
    print(f"session_id={summary.session_id}")
    print(f"session_name={summary.session_name}")
    print(f"observability_mode={summary.observability_mode}")
    print(f"turns_requested={summary.turns_requested}")
    print(f"turns_completed={summary.turns_completed}")
    print(f"stop_reason={summary.stop_reason}")
    print(f"error_count={summary.error_count}")
    print(f"text_only_turns={summary.text_only_turns}")
    print(f"elapsed_seconds={summary.elapsed_seconds:.2f}")
    for agent in sorted(summary.per_agent_response_statuses):
        statuses = summary.per_agent_response_statuses[agent]
        done_count = summary.per_agent_response_done_count[agent]
        print(
            f"{agent}: response_done={done_count}, statuses={statuses if statuses else []}"
        )
    print("turn_results:")
    for turn in summary.turn_results:
        print(
            f"  turn={turn.turn_index} speaker={turn.speaker_label} status={turn.response_status} "
            f"audio_bytes={turn.audio_bytes} relayed={turn.relayed} text_only={turn.text_only} "
            f"transcriptions={turn.transcription_events} transcript={repr(turn.assistant_transcript)} "
            f"tool_calls={turn.tool_calls} stop_call={{'id': {repr(turn.stop_call_call_id)}, "
            f"'outcome': {repr(turn.stop_call_outcome)}, 'reason': {repr(turn.stop_call_reason)}}} "
            f"errors={turn.errors}"
        )


def _parse_args() -> tuple[AutomationConfig, bool, list[str]]:
    scenario_keys = sorted(SCENARIOS.keys())
    parser = argparse.ArgumentParser(
        description="Automated OpenAI Realtime A<->B voice relay demo for Maxim."
    )
    parser.add_argument("--turns", type=int, default=AutomationConfig.turns)
    parser.add_argument("--model", type=str, default=AutomationConfig.model)
    parser.add_argument(
        "--scenario",
        type=str,
        default=AutomationConfig.scenario,
        choices=scenario_keys,
        help=f"Simulation scenario. Available: {', '.join(scenario_keys)}",
    )
    parser.add_argument(
        "--run-scenario-loop",
        action="store_true",
        help="Run multiple scenarios sequentially (one session per scenario).",
    )
    parser.add_argument(
        "--loop-scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIO_LOOP),
        help=(
            "Comma-separated scenario keys for loop mode. "
            f"Default: {','.join(DEFAULT_SCENARIO_LOOP)}"
        ),
    )
    parser.add_argument("--voice-a", type=str, default=AutomationConfig.voice_a)
    parser.add_argument("--voice-b", type=str, default=AutomationConfig.voice_b)
    parser.add_argument(
        "--bootstrap-instructions",
        type=str,
        default=None,
        help="Optional bootstrap prompt override. If omitted, scenario default is used.",
    )
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--turn-timeout-sec", type=float, default=45.0)
    parser.add_argument(
        "--session-name",
        type=str,
        default=AutomationConfig.session_name,
    )
    parser.add_argument("--debug-events", action="store_true")
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument(
        "--observability-mode",
        type=str,
        default=AutomationConfig.observability_mode,
        choices=["dual", "single-agent-a", "single-agent-b"],
        help=(
            "Observability mode: dual logs both sides, single-agent-a logs only "
            "agent A, single-agent-b logs only agent B."
        ),
    )
    parser.add_argument(
        "--single-observed-agent",
        action="store_true",
        help=(
            "Shortcut for --observability-mode single-agent-b "
            "(typical production perspective)."
        ),
    )
    args = parser.parse_args()

    observability_mode = (
        "single-agent-b" if args.single_observed_agent else args.observability_mode
    )

    config = AutomationConfig(
        turns=args.turns,
        model=args.model,
        scenario=args.scenario,
        voice_a=args.voice_a,
        voice_b=args.voice_b,
        bootstrap_instructions=args.bootstrap_instructions,
        chunk_ms=args.chunk_ms,
        turn_timeout_sec=args.turn_timeout_sec,
        session_name=args.session_name,
        debug_events=args.debug_events,
        max_failures=args.max_failures,
        observability_mode=observability_mode,
    )
    loop_scenarios = [x.strip() for x in args.loop_scenarios.split(",") if x.strip()]
    if args.run_scenario_loop:
        invalid = [x for x in loop_scenarios if x not in SCENARIOS]
        if invalid:
            raise ValueError(
                f"Invalid --loop-scenarios values: {invalid}. "
                f"Allowed: {sorted(SCENARIOS.keys())}"
            )
    return config, bool(args.run_scenario_loop), loop_scenarios


async def _main_async() -> None:
    config, run_scenario_loop, loop_scenarios = _parse_args()
    if not run_scenario_loop:
        summary = await run_duplex_voice_automation(config)
        _print_summary(summary)
        return

    shared_logger, shared_client, shared_plain_client = (
        _build_shared_logger_and_clients()
    )
    summaries: list[AutomationRunSummary] = []
    total = len(loop_scenarios)
    print(f"\n=== Scenario Loop: {total} runs ===")
    for idx, scenario_key in enumerate(loop_scenarios, start=1):
        scenario = _get_scenario_profile(scenario_key)
        print(f"\n--- Run {idx}/{total}: {scenario.title} ({scenario_key}) ---")
        run_config = replace(config, scenario=scenario_key)
        summary = await run_duplex_voice_automation(
            run_config,
            logger=shared_logger,
            client_aio=shared_client,
            plain_client_aio=shared_plain_client,
            flush_logger=True,
        )
        _print_summary(summary)
        summaries.append(summary)

    # Ensure all logs are flushed when running loop mode with shared logger.
    shared_logger.cleanup(is_sync=True)

    total_turns = sum(s.turns_completed for s in summaries)
    total_errors = sum(s.error_count for s in summaries)
    print("\n=== Scenario Loop Aggregate ===")
    print(f"runs={len(summaries)}")
    print(f"turns_completed_total={total_turns}")
    print(f"errors_total={total_errors}")
    print("session_ids:")
    for summary in summaries:
        print(f"  {summary.session_id} ({summary.session_name})")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
