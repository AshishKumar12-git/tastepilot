
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://hackathon.api.qloo.com"

INSIGHTS_ENDPOINT = f"{BASE_URL}/v2/insights"
TAGS_ENDPOINT = f"{BASE_URL}/v2/tags"

TIMEOUT_SECONDS = 30


class QlooAPIError(Exception):
    """Raised when a Qloo API request fails."""


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

def get_api_key():
    """Read the Qloo API key from environment variables."""

    api_key = os.getenv("QLOO_API_KEY")

    if not api_key:
        raise QlooAPIError(
            "QLOO_API_KEY is missing. Check your .env file."
        )

    return api_key


def _get_headers():
    """Build common Qloo API headers."""

    return {
        "X-Api-Key": get_api_key(),
        "Accept": "application/json",
    }


# ---------------------------------------------------------
# Parameter normalization
# ---------------------------------------------------------

def normalize_params(params):
    """
    Normalize parameters before sending GET requests.

    Lists of tag IDs must be comma-separated strings,
    not repeated query parameters.
    """

    if not isinstance(params, dict):
        raise ValueError(
            "params must be a dictionary."
        )

    normalized = {}

    for key, value in params.items():

        if value is None:
            continue

        if isinstance(value, (list, tuple)):

            if not value:
                continue

            normalized[key] = ",".join(
                str(item) for item in value
            )

        else:
            normalized[key] = value

    return normalized


# ---------------------------------------------------------
# Shared GET request handler
# ---------------------------------------------------------

def _send_get_request(url, params):
    """
    Shared GET request handler for Qloo endpoints.

    Handles:
        - Authentication
        - Parameter normalization
        - Timeouts
        - HTTP errors
        - Invalid JSON
        - Unexpected response formats
    """

    normalized_params = normalize_params(params)

    try:
        response = requests.get(
            url,
            params=normalized_params,
            headers=_get_headers(),
            timeout=TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.Timeout as exc:

        raise QlooAPIError(
            f"Qloo request timed out: {url}"
        ) from exc

    except requests.exceptions.HTTPError as exc:

        status = exc.response.status_code
        body = exc.response.text[:1000]

        raise QlooAPIError(
            f"Qloo returned HTTP {status}.\n"
            f"Endpoint: {url}\n"
            f"Response: {body}"
        ) from exc

    except requests.exceptions.JSONDecodeError as exc:

        raise QlooAPIError(
            f"Qloo returned invalid JSON.\n"
            f"Endpoint: {url}"
        ) from exc

    except requests.exceptions.RequestException as exc:

        raise QlooAPIError(
            f"Could not connect to Qloo.\n"
            f"Endpoint: {url}\n"
            f"Error: {exc}"
        ) from exc

    if not isinstance(data, dict):

        raise QlooAPIError(
            "Qloo returned an unexpected response format."
        )

    if data.get("success") is False:

        raise QlooAPIError(
            f"Qloo reported that the request failed: {data}"
        )

    return data


# ---------------------------------------------------------
# Insights API
# ---------------------------------------------------------

def get_insights(params):
    """
    Generic Qloo Insights API client.

    Accepts parameters prepared by qloo_mapper.py.

    No category-specific decisions are made here.
    """

    if not isinstance(params, dict):
        raise ValueError(
            "params must be a dictionary."
        )

    if not params:
        raise ValueError(
            "params cannot be empty."
        )

    return _send_get_request(
        INSIGHTS_ENDPOINT,
        params,
    )


# ---------------------------------------------------------
# Tags API
# ---------------------------------------------------------

def search_qloo_tags(
    query,
    limit=10,
    extra_params=None,
):
    """
    Search Qloo's tag catalogue.

    Supports optional query parameters such as:
        filter.tag.types

    Used by tag_resolver.py to discover and validate tags.
    """

    if not isinstance(query, str) or not query.strip():

        raise ValueError(
            "query must be a non-empty string."
        )

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
    ):

        raise ValueError(
            "limit must be a positive integer."
        )

    params = {
        "filter.query": query.strip(),
        "take": limit,
    }

    if extra_params is not None:

        if not isinstance(extra_params, dict):

            raise TypeError(
                "extra_params must be a dictionary."
            )

        params.update(extra_params)

    return _send_get_request(
        TAGS_ENDPOINT,
        params,
    )
