HIGH_KEYWORDS = [
    "critical", "outage", "down", "failed", "fatal",
    "exceeded threshold", "unavailable", "crash", "oom",
]
MEDIUM_KEYWORDS = [
    "warning", "slow", "degraded", "latency",
    "timeout", "retry", "elevated", "error",
]

_SOURCE_BASE_PRIORITY = {
    "alert": "medium",
    "ticket": "low",
    "log": "low",
}

_PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_PRIORITY = {v: k for k, v in _PRIORITY_RANK.items()}

def _keyword_priority(text: str) -> tuple[str, float]:
    text_lower = text.lower()

    high_hits = sum(1 for kw in HIGH_KEYWORDS if kw in text_lower)
    medium_hits = sum(1 for kw in MEDIUM_KEYWORDS if kw in text_lower)

    total_weight = high_hits * 1.0 + medium_hits * 0.5
    max_possible = max(len(HIGH_KEYWORDS) * 1.0 + len(MEDIUM_KEYWORDS) * 0.5, 1)
    score = min(total_weight / max_possible, 1.0)

    if high_hits > 0:
        return "high", score
    if medium_hits > 0:
        return "medium", score
    return "low", score


def score_priority(text: str, source_type: str) -> dict:
    base = _SOURCE_BASE_PRIORITY.get(source_type, "low")
    keyword_priority, score = _keyword_priority(text)

    #text: "Retry attempt 2 for DNS lookup"
    #source_type: "log"
    #base → low → rank 0
    #keyword_priority → medium → rank 1 ("retry" is a medium keyword)
    #max(0, 1) = 1 → "medium"
    #So a log entry (normally low) gets bumped to medium because the text contains a medium keyword. The source type sets a floor and keywords can only push the priority up, never down.
    final_rank = max(_PRIORITY_RANK[base], _PRIORITY_RANK[keyword_priority])
    final_priority = _RANK_PRIORITY[final_rank]

    return {"priority": final_priority, "score": round(score, 4)}


def escalate_to_high(incidents: list[dict]) -> list[dict]:
    #This function escalates all incidents in a cluster to high if they come from at least 2 different source types.
    source_types = {inc["source_type"] for inc in incidents}
    if len(source_types) >= 2:
        #Returns a new list where every incident dict is copied (**inc) but with priority overwritten to "high".
        #If only one source type is present (e.g. 3 logs, no ticket or alert), no escalation - return unchanged.
        return [{**inc, "priority": "high"} for inc in incidents]
    return incidents


if __name__ == "__main__":
    cases = [
        ("CRITICAL: auth service outage", "alert"),
        ("Retry attempt 2 for DNS lookup", "log"),
        ("Scheduled maintenance window", "ticket"),
        ("Token validation failed for user admin", "log"),
        ("Pod restarting repeatedly. OOMKilled in kubelet.", "alert"),
    ]
    for text, source in cases:
        result = score_priority(text, source)
        print(f"[{source:6}] {text[:55]:<55} -> {result['priority']:6} (score={result['score']})")
