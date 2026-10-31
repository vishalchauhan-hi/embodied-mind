"""
VLM-based visual perceiver for grounding raw game frames into structured scene descriptions.
Supports Google Gemini, OpenAI GPT-4o, and local models (LLaVA via Ollama).
"""

import os
import json
import base64
import logging
from io import BytesIO
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

OFFLINE_MODELS = {"rule", "mock", "offline"}
ANIMALS = {"sheep", "cow", "pig", "chicken", "zombie", "skeleton", "creeper", "spider"}


def _gemini_client():
    """Return a Gemini client, preferring the current google-genai SDK."""
    try:
        from google import genai  # google-genai (current SDK)
        return ("genai", genai.Client(api_key=os.getenv("GOOGLE_API_KEY")))
    except ImportError:
        import google.generativeai as legacy  # google-generativeai (older SDK)
        legacy.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return ("legacy", legacy)

PERCEPTION_PROMPT = (Path(__file__).parent.parent / "prompts" / "perceiver.txt").resolve()

DEFAULT_PROMPT = """You are a visual perception module for an embodied AI agent in Minecraft.
Analyze the given screenshot and extract structured information.

Respond ONLY with a JSON object (no markdown, no backticks):
{
  "summary": "One-sentence description of what the agent sees",
  "entities": [
    {"type": "cow/pig/zombie/skeleton/etc", "distance": "near/medium/far", "direction": "left/center/right"}
  ],
  "resources": [
    {"type": "wood/stone/iron_ore/etc", "distance": "near/medium/far", "quantity": "few/some/many"}
  ],
  "terrain": {
    "biome": "plains/forest/desert/mountain/etc",
    "ground": "grass/dirt/sand/stone/snow",
    "obstacles": ["water", "cliff", "lava"],
    "time_of_day": "day/sunset/night"
  },
  "threats": [
    {"type": "zombie/skeleton/creeper/fall/lava", "severity": "low/medium/high", "distance": "near/medium/far"}
  ],
  "actionable_items": ["list of things the agent could interact with right now"]
}
"""


class VLMPerceiver:
    """Converts raw RGB frames into structured scene descriptions using a VLM."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.prompt = self._load_prompt()
        self._init_client()
        logger.info(f"VLMPerceiver initialized | model={model}")

    def _load_prompt(self) -> str:
        if PERCEPTION_PROMPT.exists():
            return PERCEPTION_PROMPT.read_text()
        return DEFAULT_PROMPT

    def _init_client(self):
        """Initialize the appropriate API client based on model name."""
        if self.model in OFFLINE_MODELS:
            self.client = None
            self.api = "rule"
        elif "gemini" in self.model:
            self.client, self.api = _gemini_client(), "gemini"
        elif "gpt" in self.model:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.api = "openai"
        elif "llava" in self.model or "ollama" in self.model:
            import requests
            self.client = requests  # Use requests for Ollama API
            self.api = "ollama"
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def perceive(self, rgb_frame: np.ndarray, obs: Optional[dict] = None) -> dict:
        """
        Process a raw RGB frame and return a structured scene description.

        Args:
            rgb_frame: numpy array of shape (H, W, 3) with uint8 values
            obs: full observation, used only by the offline rule perceiver

        Returns:
            dict with keys: summary, entities, resources, terrain, threats, actionable_items
        """
        if self.api == "rule":
            return self._rule_perception(obs or {})

        img = Image.fromarray(rgb_frame)
        img_b64 = self._encode_image(img)

        try:
            response_text = self._call_vlm(img_b64, img)
            perception = self._parse_response(response_text)
            return perception
        except Exception as e:
            logger.warning(f"Perception failed: {e}. Returning fallback.")
            return self._fallback_perception()

    def _rule_perception(self, obs: dict) -> dict:
        """Offline perception: reads the mock env's ground-truth visibility list."""
        visible = obs.get("mock_visible", [])
        if not visible:
            p = self._fallback_perception()
            p["summary"] = "Nothing of interest in view"
            return p
        entities = [v for v in visible if v["type"] in ANIMALS]
        resources = [v for v in visible if v["type"] not in ANIMALS]
        names = ", ".join(f"{v['type']} ({v['distance']}, {v['direction']})" for v in visible)
        return {
            "summary": f"Visible: {names}",
            "entities": entities,
            "resources": resources,
            "terrain": {"biome": "plains", "ground": "grass", "obstacles": [], "time_of_day": "day"},
            "threats": [],
            "actionable_items": [v["type"] for v in visible if v["distance"] == "near"],
        }

    def _encode_image(self, img: Image.Image, max_size: int = 512) -> str:
        """Resize and encode image to base64."""
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_vlm(self, img_b64: str, img: Image.Image) -> str:
        """Route to the appropriate VLM API."""
        if self.api == "gemini":
            return self._call_gemini(img)
        elif self.api == "openai":
            return self._call_openai(img_b64)
        elif self.api == "ollama":
            return self._call_ollama(img_b64)

    def _call_gemini(self, img: Image.Image) -> str:
        kind, client = self.client
        if kind == "genai":
            response = client.models.generate_content(model=self.model, contents=[self.prompt, img])
        else:
            response = client.GenerativeModel(self.model).generate_content([self.prompt, img])
        return response.text

    def _call_openai(self, img_b64: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }},
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def _call_ollama(self, img_b64: str) -> str:
        response = self.client.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model.replace("ollama/", ""),
                "prompt": self.prompt,
                "images": [img_b64],
                "stream": False,
            },
        )
        return response.json()["response"]

    def _parse_response(self, text: str) -> dict:
        """Parse JSON response from VLM, handling common formatting issues."""
        cleaned = text.strip()
        # Strip markdown code fences
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            import re
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse VLM response: {cleaned[:200]}")

        # Validate expected keys
        for key in ["summary", "entities", "resources", "terrain", "threats"]:
            if key not in result:
                result[key] = [] if key != "summary" and key != "terrain" else {}
        if "summary" not in result:
            result["summary"] = "No description available"
        if "terrain" not in result or not isinstance(result["terrain"], dict):
            result["terrain"] = {"biome": "unknown", "ground": "unknown",
                                  "obstacles": [], "time_of_day": "unknown"}

        return result

    def _fallback_perception(self) -> dict:
        """Return a minimal perception when VLM fails."""
        return {
            "summary": "Unable to process visual input",
            "entities": [],
            "resources": [],
            "terrain": {"biome": "unknown", "ground": "unknown",
                        "obstacles": [], "time_of_day": "unknown"},
            "threats": [],
            "actionable_items": [],
        }
