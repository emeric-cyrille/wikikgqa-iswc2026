"""Thin SPARQL client for the WikiKGQA QLever endpoint.

The endpoint requires PREFIX declarations for the Wikidata namespaces (no implicit
prefixes). This module prepends a standard prefix header to every query.

Read access is open (no auth needed). Optional credentials (env vars
WIKIKGQA_USER/WIKIKGQA_PASS) are forwarded as HTTP Basic if both are set.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from base64 import b64encode

ENDPOINT = os.environ.get(
    "WIKIKGQA_ENDPOINT", "https://backend.wikikgqa.skynet.coypu.org"
)

PREFIXES = """\
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wds: <http://www.wikidata.org/entity/statement/>
PREFIX wdv: <http://www.wikidata.org/value/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
PREFIX pr: <http://www.wikidata.org/prop/reference/>
PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
"""


def _build_request(query: str, timeout_s: int) -> urllib.request.Request:
    full = PREFIXES + "\n" + query
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": full})
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/sparql-results+json")
    u = os.environ.get("WIKIKGQA_USER")
    p = os.environ.get("WIKIKGQA_PASS")
    if u and p:
        token = b64encode(f"{u}:{p}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    return req


class SPARQLError(RuntimeError):
    pass


def run_sparql(query: str, timeout_s: int = 30, retries: int = 3) -> dict:
    """Execute a SPARQL query and return the raw SPARQL JSON Results object.

    Raises SPARQLError on HTTP error or QLever-reported query error. Transient
    network errors (DNS hiccup, connection reset, gateway timeout) are retried
    with exponential backoff.
    """
    import time as _time
    body: str | None = None
    for attempt in range(retries):
        req = _build_request(query, timeout_s)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode()
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                _time.sleep(min(2 ** attempt + 1, 30))
                continue
            raise SPARQLError(f"HTTP {e.code}: {err_body[:500]}") from e
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                _time.sleep(min(2 ** attempt + 1, 30))
                continue
            raise SPARQLError(f"Network error: {e}") from e
    if body is None:
        raise SPARQLError("Network error: no response after retries")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise SPARQLError(f"Non-JSON response: {body[:500]}") from e
    if isinstance(data, dict) and data.get("status") == "ERROR":
        raise SPARQLError(data.get("exception", "Unknown QLever error"))
    return data


if __name__ == "__main__":
    # quick smoke test
    sample = "SELECT DISTINCT ?obj2 WHERE { wd:Q19798734 p:P161 ?obj1 . ?obj1 pq:P453 wd:Q27955792 . ?obj1 ps:P161 ?obj2 . }"
    print(json.dumps(run_sparql(sample), indent=2))
