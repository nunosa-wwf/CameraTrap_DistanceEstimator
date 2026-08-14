"""
Optional AI-generated summary report, using the person's own Anthropic API key.

The key is only ever held in memory for the duration of the app session --
it is never written to disk, logged, or included in any exported file. It's
re-entered each time the app is launched.

Requires internet access to api.anthropic.com from the machine running the
app (this is separate from -- and does not affect -- the local depth model,
which runs fully offline once its weights are cached).
"""

import json
import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"


def generate_report(api_key, context: dict) -> str:
    """context is a dict of whatever's available: tie point count, calibration
    method + MAE, uncertainty, target distance, suspected species, location, etc.
    Missing fields are fine -- just omit them from the dict."""

    if not api_key or not api_key.strip():
        return "No API key provided -- enter your Anthropic API key above to generate a report."

    context_lines = []
    for key, val in context.items():
        if val not in (None, ""):
            context_lines.append(f"- {key}: {val}")
    context_block = "\n".join(context_lines) if context_lines else "(no data provided)"

    prompt = f"""You are helping a field ecologist sanity-check the output of a camera-trap
distance-estimation pipeline (monocular depth model + field-measured tie-point calibration).

Session data:
{context_block}

Write a short (4-6 paragraph) plain-language report covering:
1. Calibration quality: given the tie point count and leave-one-out error, how much should
   the operator trust the corrected distance estimate? Be specific about what the MAE/
   uncertainty numbers mean in practical terms (e.g. "expect the true distance to typically
   fall within +/- X m of the reported value").
2. Distance plausibility: is the reported distance typical/atypical for a camera-trap
   detection, given anything stated about the animal or setup?
3. Species/location plausibility, ONLY if both a suspected species and a location were
   given: is this species plausible in this general region/habitat, based on your general
   knowledge? Be explicit that this is a general-knowledge sanity check, not a verified
   range-map lookup, and the operator should confirm against actual regional species lists
   if the answer matters for their work.
4. One or two concrete suggestions for improving calibration confidence next time, if
   anything in the data suggests a specific weakness (e.g. few tie points, high uncertainty,
   collinear tie points).

If a field wasn't provided, don't speculate about it -- just skip that part of the analysis.
Keep the tone practical and calibrated, not alarmist or overconfident."""

    try:
        resp = requests.post(
            API_URL,
            headers={
                "x-api-key": api_key.strip(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            data=json.dumps({
                "model": MODEL,
                "max_tokens": 1200,
                "messages": [{"role": "user", "content": prompt}],
            }),
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        return f"Could not reach the Anthropic API -- check your internet connection.\n\nDetails: {e}"

    if resp.status_code == 401:
        return "API key rejected (401 Unauthorized). Double-check the key and try again."
    if resp.status_code != 200:
        return f"API request failed (status {resp.status_code}).\n\n{resp.text[:500]}"

    data = resp.json()
    try:
        text_parts = [block["text"] for block in data["content"] if block.get("type") == "text"]
        return "\n".join(text_parts).strip()
    except (KeyError, IndexError):
        return f"Unexpected response format from API:\n\n{json.dumps(data, indent=2)[:800]}"
