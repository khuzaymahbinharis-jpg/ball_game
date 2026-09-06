import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import httpx
from dotenv import load_dotenv
from PIL import Image

from .schema import FrameDetection, frame_detection_json_schema


DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-lite"
PROMPT_VERSION = "test1-ruler-v1"


class VLMProvider(Protocol):
    def detect(self, frame_id: int, image: Image.Image) -> FrameDetection: ...


class OpenRouterError(RuntimeError):
    """A safe provider error that never contains a credential or request headers."""


class OpenRouterValidationError(OpenRouterError):
    def __init__(
        self,
        message: str,
        raw_content: str | None,
        usage: "ProviderUsage",
        latency_seconds: float,
        request_id: str | None,
        returned_model: str | None,
        provider: str | None,
        raw_response_body: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.raw_content = raw_content
        self.usage = usage
        self.latency_seconds = latency_seconds
        self.request_id = request_id
        self.returned_model = returned_model
        self.provider = provider
        self.raw_response_body = raw_response_body


@dataclass(frozen=True)
class ProviderUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None


@dataclass(frozen=True)
class OpenRouterResult:
    detection: FrameDetection
    request_id: str | None
    returned_model: str | None
    provider: str | None
    latency_seconds: float
    usage: ProviderUsage
    raw_response_body: dict[str, Any] | None = None


def detection_prompt(frame_id: int) -> str:
    return f"""You are the semantic visual detector for one basketball video frame.

Return only the requested structured JSON. The frame_id must be {frame_id}.

The image has an artificial horizontal ruler above the original content and a
vertical ruler to its left. Every coordinate in the JSON must use the rulers'
normalized 0.0-1.0 coordinate system relative to the ORIGINAL VIDEO CONTENT.
Do not include the white ruler margins in any coordinate.

Detect every visible on-court player, including partially occluded players, but
exclude referees, spectators, coaches, bench personnel, and graphics. Assign the
two visually distinct uniform groups as Team A and Team B consistently within
this frame. Use "uncertain" only when the uniform cannot be assigned reliably.
Give each player a unique local detection_id such as player_01. For each player:
- box is a tight normalized body bounding box (left, top, right, bottom);
- foot is the normalized ground/contact point centered between the feet, or the
  best estimated ground point when the feet are occluded;
- confidence covers player localization and team_confidence covers team label.

For the basketball, return a normalized center and a tight box when practical.
Use null for ball only when it cannot be located. Return possession only when a
specific returned player likely controls the ball; otherwise use null. Put short,
concrete ambiguities in uncertainty_notes. Never invent invisible objects."""


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _usage_from_body(body: dict[str, Any]) -> ProviderUsage:
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    return ProviderUsage(
        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
        completion_tokens=_optional_int(usage.get("completion_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        reasoning_tokens=_optional_int(completion_details.get("reasoning_tokens")),
        cost_usd=_optional_float(usage.get("cost")),
    )


class OpenRouterVLMProvider:
    """Single-request OpenRouter adapter. It performs no automatic retries."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENROUTER_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        reasoning_effort: str = "minimal",
        max_output_tokens: int = 4096,
        timeout_seconds: float = 45.0,
        client: httpx.Client | None = None,
        reasoning_setting: dict[str, Any] | None = None,
        include_reasoning_parameter: bool = True,
        provider_preferences: dict[str, Any] | None = None,
    ):
        if not api_key.strip():
            raise ValueError("OpenRouter API key is empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._client = client
        self.reasoning_setting = (
            {"effort": reasoning_effort, "exclude": True}
            if reasoning_setting is None
            else dict(reasoning_setting)
        )
        self.include_reasoning_parameter = include_reasoning_parameter
        self.provider_preferences = (
            None if provider_preferences is None else dict(provider_preferences)
        )

    @classmethod
    def from_repository_env(
        cls, repository_root: str | Path, **kwargs: Any
    ) -> "OpenRouterVLMProvider":
        env_path = Path(repository_root) / ".env"
        if not env_path.is_file():
            raise FileNotFoundError(
                f"Create {env_path} with OPENROUTER_API_KEY before an approved run"
            )
        load_dotenv(dotenv_path=env_path, override=False)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing from the environment")
        return cls(api_key=api_key, **kwargs)

    def request_detection(
        self, frame_id: int, image: Image.Image, prompt: str | None = None
    ) -> OpenRouterResult:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt if prompt is not None else detection_prompt(frame_id),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image)},
                        },
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "frame_detection",
                    "strict": True,
                    "schema": frame_detection_json_schema(),
                },
            },
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
        }
        if self.include_reasoning_parameter:
            payload["reasoning"] = self.reasoning_setting
        if self.provider_preferences is not None:
            payload["provider"] = self.provider_preferences
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "Zeta Track the Game Test 1",
        }
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        owns_client = self._client is None
        started = perf_counter()
        try:
            response = client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            latency = perf_counter() - started
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise OpenRouterError(
                f"OpenRouter returned HTTP {exc.response.status_code}; no retry was made"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise OpenRouterError("OpenRouter request failed; no retry was made") from exc
        finally:
            if owns_client:
                client.close()

        usage = _usage_from_body(body)
        request_id = body.get("id") if isinstance(body.get("id"), str) else None
        returned_model = body.get("model") if isinstance(body.get("model"), str) else None
        returned_provider = body.get("provider") if isinstance(body.get("provider"), str) else None
        content = None
        try:
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content was not text")
            parsed = json.loads(content)
            detection = FrameDetection.from_dict(parsed)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OpenRouterValidationError(
                "OpenRouter response failed schema validation",
                content if isinstance(content, str) else None,
                usage,
                latency,
                request_id,
                returned_model,
                returned_provider,
                body,
            ) from exc
        if detection.frame_id != frame_id:
            raise OpenRouterValidationError(
                "OpenRouter response returned the wrong frame_id",
                content,
                usage,
                latency,
                request_id,
                returned_model,
                returned_provider,
                body,
            )
        return OpenRouterResult(
            detection=detection,
            request_id=request_id,
            returned_model=returned_model,
            provider=returned_provider,
            latency_seconds=latency,
            usage=usage,
            raw_response_body=body,
        )

    def detect(self, frame_id: int, image: Image.Image) -> FrameDetection:
        return self.request_detection(frame_id, image).detection


class MockVLMProvider:
    """Deterministic fixture-like provider. It does no image interpretation."""

    def detect(self, frame_id: int, image: Image.Image) -> FrameDetection:
        del image
        shift = min(frame_id * 0.005, 0.25)
        return FrameDetection.from_dict(
            {
                "frame_id": frame_id,
                "players": [
                    {
                        "detection_id": f"{frame_id}-a",
                        "team": "A",
                        "box": {
                            "left": 0.1 + shift,
                            "top": 0.3,
                            "right": 0.18 + shift,
                            "bottom": 0.75,
                        },
                        "foot": {"x": 0.14 + shift, "y": 0.75},
                        "confidence": 0.94,
                        "team_confidence": 0.98,
                    },
                    {
                        "detection_id": f"{frame_id}-b",
                        "team": "B",
                        "box": {
                            "left": 0.68 - shift,
                            "top": 0.28,
                            "right": 0.76 - shift,
                            "bottom": 0.72,
                        },
                        "foot": {"x": 0.72 - shift, "y": 0.72},
                        "confidence": 0.91,
                        "team_confidence": 0.97,
                    },
                ],
                "ball": {
                    "center": {"x": 0.2 + shift, "y": 0.72},
                    "box": None,
                    "confidence": 0.88,
                },
                "possession": {
                    "player_detection_id": f"{frame_id}-a",
                    "confidence": 0.84,
                },
                "uncertainty_notes": [],
            }
        )
