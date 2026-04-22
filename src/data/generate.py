import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42
OUTPUT_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"

HIGH_KEYWORDS = ["critical", "outage", "down", "failed", "fatal",
                 "exceeded threshold", "unavailable", "crash", "oom"]
MEDIUM_KEYWORDS = ["warning", "slow", "degraded", "latency",
                   "timeout", "retry", "elevated", "error"]

TICKET_TEMPLATES = {
    "network_issue": [
        "Users reporting packet loss on corporate network. DNS resolution failing intermittently.",
        "VPN connectivity dropped for remote workers. Firewall blocking outbound traffic on port 443.",
        "High latency observed on backbone switch. Network degraded across floor 3.",
        "DNS lookup failures affecting internal services. Routing table misconfiguration suspected.",
        "Intermittent packet loss detected on WAN link. Throughput reduced by 60%.",
        "Firewall rule blocking traffic between VLAN 10 and VLAN 20. Services unreachable.",
        "Network switch failover not triggering. Redundant path unavailable.",
        "BGP route propagation delay causing intermittent connectivity failures.",
        "Users unable to reach external services. NAT translation failing on edge router.",
        "Load balancer reporting upstream connection timeouts. Backend nodes unreachable.",
    ],
    "authentication_failure": [
        "Users unable to authenticate to VPN. Error: token validation failed.",
        "SSO login broken for sales team. JWT expired errors in browser console.",
        "Active Directory authentication failing for contractor accounts.",
        "MFA push notifications not delivered. Users locked out of corporate apps.",
        "OAuth2 token endpoint returning 401 for service accounts.",
        "LDAP bind operation failing. Directory service returning connection refused.",
        "Kerberos ticket renewal broken after patch deployment.",
        "API gateway rejecting valid Bearer tokens. Auth service not responding.",
        "User sessions expiring prematurely. Session store connectivity issue.",
        "Password reset emails not sent. SMTP relay authentication rejected.",
    ],
    "deployment_issue": [
        "Production pod restarting repeatedly after latest release. OOM kill in logs.",
        "Deployment rollback triggered for payments service. Health check failing.",
        "Container image pull failing in prod namespace. Registry credentials expired.",
        "Kubernetes node NotReady after node pool upgrade. Workloads evicted.",
        "Helm chart deployment stuck in pending state. PVC not bound.",
        "CI/CD pipeline failing at integration test stage after merge.",
        "Blue/green switch failed. Old version still serving traffic after deployment.",
        "Service mesh sidecar injection failing. Pods starting without Envoy proxy.",
        "Database migration failed during deployment. Schema version mismatch.",
        "Canary release showing elevated error rate. Automatic rollback not triggered.",
    ],
}

LOG_TEMPLATES = {
    "network_issue": [
        "ERROR network-agent PacketLossException: 34% packet loss on interface eth0 at {ts}",
        "WARN dns-resolver DNSTimeoutError: resolution failed for internal.corp after 3 retries at {ts}",
        "ERROR firewall-daemon RuleViolation: traffic blocked src=10.0.1.5 dst=10.0.2.1 port=443 at {ts}",
        "CRITICAL routing-daemon RouteFlap: BGP session dropped with peer 192.168.1.1 at {ts}",
        "ERROR switch-monitor HighLatency: round-trip time 850ms on backbone link at {ts}",
        "WARN nat-gateway TranslationFail: SNAT pool exhausted, dropping connections at {ts}",
        "ERROR load-balancer UpstreamTimeout: all backends unreachable for pool web-prod at {ts}",
        "CRITICAL vpn-gateway TunnelDown: IPSec SA expired, re-keying failed at {ts}",
        "ERROR dhcp-server PoolExhausted: no free leases in subnet 10.10.0.0/24 at {ts}",
        "WARN spanning-tree TopologyChange: STP reconverging on VLAN 100 at {ts}",
    ],
    "authentication_failure": [
        "ERROR auth-service TokenValidationException: JWT expired at {ts}",
        "ERROR sso-provider OAuthError: invalid_grant for client_id=webapp-prod at {ts}",
        "CRITICAL ldap-client BindFailure: LDAP server returned error 49 invalid credentials at {ts}",
        "ERROR kerberos-agent TGTRenewalFailed: KDC unreachable for realm CORP.LOCAL at {ts}",
        "WARN mfa-service PushTimeout: push notification undelivered for user jdoe after 30s at {ts}",
        "ERROR api-gateway AuthRejected: Bearer token signature mismatch for /api/v2/orders at {ts}",
        "CRITICAL session-store ConnectionRefused: Redis auth backend unreachable at {ts}",
        "ERROR ad-connector QueryTimeout: Active Directory LDAP query timed out after 5000ms at {ts}",
        "WARN oauth-server RateLimited: token endpoint requests exceeded 1000/min at {ts}",
        "ERROR saml-proxy AssertionExpired: SAML response timestamp out of allowed skew at {ts}",
    ],
    "deployment_issue": [
        "CRITICAL kubelet OOMKilled: container payments-api killed, limit 512Mi exceeded at {ts}",
        "ERROR helm-controller DeployFailed: rollout of payments-service v2.3.1 failed at {ts}",
        "ERROR containerd ImagePullBackOff: registry.corp.local/api:latest unauthorized at {ts}",
        "CRITICAL node-controller NodeNotReady: node prod-worker-3 failed health check at {ts}",
        "ERROR pvc-controller VolumeBindTimeout: PersistentVolumeClaim data-pvc unbound for 300s at {ts}",
        "WARN ci-runner TestFailure: integration tests failed at step db-migration-check at {ts}",
        "ERROR traffic-manager SwitchFailed: blue/green switch aborted, readiness probe failing at {ts}",
        "ERROR istio-injector SidecarFailed: Envoy proxy injection rejected for namespace prod at {ts}",
        "CRITICAL flyway MigrationFailed: schema version V43 failed on prod database at {ts}",
        "ERROR argo-rollouts CanaryFailed: error rate 12.3% exceeds threshold 5% for canary at {ts}",
    ],
}

ALERT_TEMPLATES = {
    "network_issue": [
        "ALERT: packet_loss_rate exceeded threshold 5% → {val}% over 10min window",
        "ALERT: dns_resolution_failures exceeded threshold 10/min → {val}/min sustained",
        "ALERT: network_latency_ms p99 exceeded 200ms → {val}ms on backbone link",
        "ALERT: firewall_drop_rate exceeded threshold 1% → {val}% of traffic blocked",
        "ALERT: bgp_session_count below threshold 2 → {val} active sessions detected",
        "ALERT: wan_throughput_mbps dropped below 500Mbps → {val}Mbps current",
        "ALERT: vpn_tunnel_stability below 99% → {val}% uptime over last 15min",
        "ALERT: load_balancer_5xx_rate exceeded 1% → {val}% errors from upstream",
        "ALERT: nat_translation_failure_rate exceeded 0.1% → {val}% of connections failing",
        "ALERT: dhcp_lease_utilization exceeded 90% → {val}% pool used",
    ],
    "authentication_failure": [
        "ALERT: auth_error_rate exceeded threshold 5% → {val}% over 10min window",
        "ALERT: jwt_validation_failures exceeded 50/min → {val}/min current rate",
        "ALERT: sso_login_failure_rate exceeded 10% → {val}% of attempts failing",
        "ALERT: ldap_query_timeout_rate exceeded 1% → {val}% of queries timing out",
        "ALERT: mfa_push_failure_rate exceeded 5% → {val}% of pushes undelivered",
        "ALERT: api_401_rate exceeded threshold 2% → {val}% of API calls unauthorized",
        "ALERT: session_expiry_rate exceeded 20/min → {val}/min premature expirations",
        "ALERT: kerberos_ticket_renewal_failures exceeded 10/min → {val}/min failures",
        "ALERT: oauth_token_endpoint_latency p95 exceeded 500ms → {val}ms current",
        "ALERT: active_directory_bind_failures exceeded 5/min → {val}/min detected",
    ],
    "deployment_issue": [
        "ALERT: pod_restart_count exceeded threshold 3 → {val} restarts in 10min",
        "ALERT: deployment_rollback triggered for payments-service — health check failing",
        "ALERT: container_oom_kill_rate exceeded 0 → {val} OOM kills in last 5min",
        "ALERT: node_not_ready_count exceeded 0 → {val} nodes in NotReady state",
        "ALERT: pvc_unbound_duration exceeded 60s → {val}s waiting for volume bind",
        "ALERT: ci_pipeline_failure_rate exceeded 20% → {val}% of recent runs failing",
        "ALERT: canary_error_rate exceeded threshold 5% → {val}% on canary pods",
        "ALERT: image_pull_failure_rate exceeded 0 → {val} failures in namespace prod",
        "ALERT: migration_job_status = failed for flyway-prod-migration",
        "ALERT: service_mesh_injection_failures exceeded 0 → {val} pods missing sidecar",
    ],
}

PRIORITY_MAP = {
    "network_issue": {
        "ticket": "medium",
        "log": "medium",
        "alert": "high",
    },
    "authentication_failure": {
        "ticket": "high",
        "log": "high",
        "alert": "high",
    },
    "deployment_issue": {
        "ticket": "medium",
        "log": "high",
        "alert": "medium",
    },
}


def _derive_priority(text: str, source_type: str, label: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in HIGH_KEYWORDS):
        return "high"
    if any(kw in text_lower for kw in MEDIUM_KEYWORDS):
        return "medium"
    return PRIORITY_MAP[label][source_type]


def _fill_template(template: str, ts: datetime) -> str:
    val = random.randint(15, 95)
    return template.format(ts=ts.strftime("%Y-%m-%d %H:%M:%S"), val=val)


def generate_incidents(n_groups_per_label: int = 50, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    random.seed(random_state)

    base_time = datetime(2024, 1, 15, 0, 0, 0)
    labels = ["network_issue", "authentication_failure", "deployment_issue"]
    source_types = ["ticket", "log", "alert"]

    rows = []
    incident_id = 0
    group_id = 0

    for label in labels:
        ticket_pool = TICKET_TEMPLATES[label].copy()
        log_pool = LOG_TEMPLATES[label].copy()
        alert_pool = ALERT_TEMPLATES[label].copy()

        for group_idx in range(n_groups_per_label):
            group_base_ts = base_time + timedelta(
                days=group_idx // 5,
                hours=(group_idx % 5) * 4,
                minutes=random.randint(0, 30),
            )

            ticket_text = ticket_pool[group_idx % len(ticket_pool)]
            log_ts = group_base_ts + timedelta(minutes=random.randint(1, 15))
            log_text = _fill_template(log_pool[group_idx % len(log_pool)], log_ts)
            alert_ts = group_base_ts + timedelta(minutes=random.randint(5, 30))
            alert_text = _fill_template(alert_pool[group_idx % len(alert_pool)], alert_ts)

            sources = [
                ("ticket", ticket_text, group_base_ts),
                ("log", log_text, log_ts),
                ("alert", alert_text, alert_ts),
            ]

            for source_type, text, ts in sources:
                priority = _derive_priority(text, source_type, label)
                rows.append({
                    "id": incident_id,
                    "incident_group_id": group_id,
                    "source_type": source_type,
                    "text": text,
                    "label": label,
                    "priority": priority,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                })
                incident_id += 1

            group_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df["id"] = range(len(df))
    return df


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_incidents()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows -> {OUTPUT_PATH}")
    print(df["label"].value_counts().to_string())
    print(df["priority"].value_counts().to_string())
    print(df["source_type"].value_counts().to_string())