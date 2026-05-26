from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42
OUTPUT_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"
REALISTIC_OUTPUT_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents_realistic.csv"

HIGH_KEYWORDS = ["critical", "outage", "down", "failed", "fatal",
                 "exceeded threshold", "unavailable", "crash", "oom"]
MEDIUM_KEYWORDS = ["warning", "slow", "degraded", "latency",
                   "timeout", "retry", "elevated", "error"]

SERVICES_NETWORK = [
    "edge-router-01", "core-switch-A", "wan-gateway", "dns-resolver-prod",
    "nat-gateway-eu", "vpn-concentrator", "firewall-cluster", "load-balancer-web",
    "bgp-peer-isp1", "dhcp-server-hq", "spine-switch-02", "border-router",
]
SERVICES_AUTH = [
    "auth-service", "sso-provider", "ldap-directory", "kerberos-kdc",
    "oauth2-server", "api-gateway", "session-store", "mfa-broker",
    "ad-connector", "saml-proxy", "token-service", "identity-provider",
]
SERVICES_DEPLOY = [
    "payments-api", "order-service", "inventory-svc", "notification-worker",
    "reporting-api", "search-service", "checkout-svc", "auth-worker",
    "data-pipeline", "analytics-api", "billing-service", "gateway-proxy",
]
SERVICES_DB = [
    "postgres-primary", "mysql-replica", "mongodb-cluster-01", "redis-cache",
    "elasticsearch-data", "cassandra-node-01", "pg-replica-eu", "mysql-primary",
    "redis-sentinel", "mongo-shard-02", "clickhouse-01", "timescaledb-prod",
]
SERVICES_CERT = [
    "nginx-ingress", "api-gateway-tls", "web-proxy", "cert-manager",
    "haproxy-ssl", "traefik-ingress", "cloudflare-origin", "acme-client",
    "vault-pki", "istio-citadel", "envoy-sds", "lets-encrypt-agent",
]
CLIENT_SERVICES = [
    "payments-api", "auth-service", "order-service", "checkout-svc",
    "analytics-api", "billing-service", "reporting-api", "search-service",
]
DOMAINS = [
    "api.internal", "auth.company.com", "vpn.corp.net", "portal.internal",
    "sso.company.com", "admin.internal", "api.prod.corp", "services.internal",
]
USERS = [
    "jsmith", "alopez", "mchen", "rnguyen", "dpatel", "skowalski",
    "fmüller", "tlefevre", "crossi", "abolarin", "ykim", "bsantos",
]
NAMESPACES = ["prod", "prod-eu", "prod-us", "prod-apac", "prod-core", "prod-data"]
SCHEMA_VERSIONS = [f"V{n}" for n in range(30, 80)]
CANARY_RATES = [f"{r:.1f}" for r in [6.2, 7.8, 9.1, 11.3, 12.7, 14.0, 15.5, 18.2]]
QUERY_TIMES = ["3.2s", "5.8s", "8.1s", "12.4s", "18.7s", "25.3s", "31.0s", "42.5s"]


def _rand_ip() -> str:
    return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _rand_subnet() -> str:
    return f"10.{random.randint(0,9)}.{random.randint(0,50)}.0/24"


def _rand_port() -> int:
    return random.choice([80, 443, 8080, 8443, 3306, 5432, 6379, 9200])


def _rand_vlan() -> int:
    return random.choice([10, 20, 30, 50, 100, 200, 300])


def _rand_pct(lo: int = 15, hi: int = 95) -> int:
    return random.randint(lo, hi)


TICKET_BUILDERS = {
    "network_issue": [
        lambda c: f"Users on subnet {c['subnet']} reporting packet loss. DNS resolution failing for {c['service']}.",
        lambda c: f"VPN connectivity dropped via {c['service']}. Firewall blocking outbound on port {c['port']}.",
        lambda c: f"High latency on {c['service']}. Network degraded, RTT {random.randint(300,900)}ms to gateway {c['src_ip']}.",
        lambda c: f"DNS lookup failures affecting {c['service']}. Routing misconfiguration suspected on {c['src_ip']}.",
        lambda c: f"Intermittent packet loss on WAN link via {c['service']}. Throughput reduced by {_rand_pct(40,80)}%.",
        lambda c: f"Firewall rule blocking traffic between VLAN {c['vlan']} and VLAN {c['vlan2']} via {c['service']}.",
        lambda c: f"Failover not triggering on {c['service']}. Redundant path to {c['dst_ip']} unavailable.",
        lambda c: f"BGP propagation delay on {c['service']} causing connectivity failures from {c['src_ip']}.",
        lambda c: f"Users cannot reach {c['service']}. NAT translation failing on gateway {c['dst_ip']}.",
        lambda c: f"Load balancer {c['service']} reporting upstream timeouts. Backend {c['dst_ip']} unreachable.",
    ],
    "authentication_failure": [
        lambda c: f"Users unable to authenticate via {c['service']}. Token validation failed for {c['user']}.",
        lambda c: f"SSO login broken via {c['service']}. JWT expired errors for {c['user']} in browser console.",
        lambda c: f"Active Directory auth failing via {c['service']} for user {c['user']}.",
        lambda c: f"MFA push notifications not delivered via {c['service']}. User {c['user']} locked out.",
        lambda c: f"OAuth2 token endpoint on {c['service']} returning 401 for account {c['user']}.",
        lambda c: f"LDAP bind failing on {c['service']}. Directory returning connection refused for {c['user']}.",
        lambda c: f"Kerberos ticket renewal broken on {c['service']} for user {c['user']} after patch.",
        lambda c: f"API gateway {c['service']} rejecting Bearer tokens. Auth backend unreachable for {c['user']}.",
        lambda c: f"Sessions expiring prematurely via {c['service']}. Session store issue affecting {c['user']}.",
        lambda c: f"Password reset failed on {c['service']}. SMTP relay rejecting auth for {c['user']}.",
    ],
    "deployment_issue": [
        lambda c: f"Pod {c['service']} restarting after release in {c['ns']}. OOM kill in kubelet logs.",
        lambda c: f"Deployment rollback triggered for {c['service']} in {c['ns']}. Health check failing.",
        lambda c: f"Container image pull failing for {c['service']} in {c['ns']}. Registry credentials expired.",
        lambda c: f"Kubernetes node NotReady after upgrade. Workloads for {c['service']} evicted in {c['ns']}.",
        lambda c: f"Helm chart deployment of {c['service']} stuck in pending state in {c['ns']}. PVC not bound.",
        lambda c: f"CI/CD pipeline failing at integration stage for {c['service']} after merge to {c['ns']}.",
        lambda c: f"Blue/green switch failed for {c['service']} in {c['ns']}. Old version still serving traffic.",
        lambda c: f"Sidecar injection failing for {c['service']} in {c['ns']}. Pods missing Envoy proxy.",
        lambda c: f"Database migration {c['schema']} failed during {c['service']} deployment in {c['ns']}.",
        lambda c: f"Canary {c['service']} in {c['ns']} shows {c['canary_rate']}% error rate. Rollback not triggered.",
    ],
    # Overlaps with network_issue (connection/timeout vocabulary) and deployment_issue (crash/health check)
    "database_failure": [
        lambda c: f"Database {c['service']} returning connection refused. {c['client_svc']} cannot connect to backend.",
        lambda c: f"High query latency on {c['service']}. P99 response time {c['query_time']} causing {c['client_svc']} timeouts.",
        lambda c: f"Replication lag on {c['service']} exceeded 60s. {c['client_svc']} serving stale data to users.",
        lambda c: f"Connection pool exhausted on {c['service']}. {c['client_svc']} pods failing health checks.",
        lambda c: f"Deadlock detected on {c['service']}. {c['client_svc']} write operations timing out.",
        lambda c: f"{c['service']} primary unavailable. Replica promotion failed, {c['client_svc']} degraded.",
        lambda c: f"Disk IO on {c['service']} saturated. {c['client_svc']} queries taking {c['query_time']} to respond.",
        lambda c: f"{c['service']} OOM kill. Connection count exceeded limits, {c['client_svc']} requests failing.",
        lambda c: f"Slow query on {c['service']} blocking {c['pool_size']} connections. {c['client_svc']} degraded.",
        lambda c: f"{c['service']} crash loop. {c['client_svc']} health check failing due to DB unreachable.",
    ],
    # Overlaps with authentication_failure (login/JWT/token/access vocabulary)
    "certificate_expiry": [
        lambda c: f"Users unable to login via {c['service']}. Browser reports SSL certificate error for {c['domain']}.",
        lambda c: f"HTTPS connection failing to {c['service']}. Certificate for {c['domain']} expired {c['days_expired']} days ago.",
        lambda c: f"Authentication broken for {c['user']} via {c['service']}. TLS handshake failing — cert expired.",
        lambda c: f"API calls to {c['service']} failing with SSL verification error. {c['domain']} cert not trusted.",
        lambda c: f"VPN login failing via {c['service']}. Certificate validation error for {c['user']}, cert expired.",
        lambda c: f"JWT signing certificate on {c['service']} expired. Token validation failing for {c['user']}.",
        lambda c: f"{c['service']} presenting expired cert for {c['domain']}. Users report access denied errors.",
        lambda c: f"mTLS authentication failed on {c['service']}. Client cert for {c['domain']} expired {c['days_expired']} days ago.",
        lambda c: f"SSO login broken via {c['service']}. SAML certificate for {c['domain']} no longer valid.",
        lambda c: f"User {c['user']} cannot authenticate via {c['service']}. TLS cert for {c['domain']} expired.",
    ],
}

LOG_BUILDERS = {
    "network_issue": [
        lambda c, ts: f"ERROR {c['service']} PacketLossException: {_rand_pct(20,60)}% loss on {c['src_ip']} at {ts}",
        lambda c, ts: f"WARN {c['service']} DNSTimeoutError: resolution failed for {c['dst_ip']} after 3 retries at {ts}",
        lambda c, ts: f"ERROR {c['service']} RuleViolation: traffic blocked src={c['src_ip']} dst={c['dst_ip']} port={c['port']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} RouteFlap: BGP session dropped with peer {c['dst_ip']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} HighLatency: RTT {random.randint(400,900)}ms to {c['dst_ip']} at {ts}",
        lambda c, ts: f"WARN {c['service']} TranslationFail: SNAT pool exhausted on subnet {c['subnet']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} UpstreamTimeout: backend {c['dst_ip']} unreachable at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} TunnelDown: IPSec SA expired for {c['src_ip']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} PoolExhausted: no free leases in {c['subnet']} at {ts}",
        lambda c, ts: f"WARN {c['service']} TopologyChange: STP reconverging on VLAN {c['vlan']} at {ts}",
    ],
    "authentication_failure": [
        lambda c, ts: f"ERROR {c['service']} TokenValidationException: JWT expired for user {c['user']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} OAuthError: invalid_grant for {c['user']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} BindFailure: LDAP error 49 invalid credentials for {c['user']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} TGTRenewalFailed: KDC unreachable for {c['user']} at {ts}",
        lambda c, ts: f"WARN {c['service']} PushTimeout: MFA push undelivered to {c['user']} after 30s at {ts}",
        lambda c, ts: f"ERROR {c['service']} AuthRejected: Bearer token mismatch for {c['user']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} ConnectionRefused: Redis auth backend unreachable for {c['user']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} QueryTimeout: AD LDAP query timed out for {c['user']} at {ts}",
        lambda c, ts: f"WARN {c['service']} RateLimited: token endpoint flooded by requests for {c['user']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} AssertionExpired: SAML response out of skew for {c['user']} at {ts}",
    ],
    "deployment_issue": [
        lambda c, ts: f"CRITICAL kubelet OOMKilled: container {c['service']} killed in {c['ns']} at {ts}",
        lambda c, ts: f"ERROR helm-controller DeployFailed: rollout of {c['service']} failed in {c['ns']} at {ts}",
        lambda c, ts: f"ERROR containerd ImagePullBackOff: {c['service']}:latest unauthorized in {c['ns']} at {ts}",
        lambda c, ts: f"CRITICAL node-controller NodeNotReady: node hosting {c['service']} failed check at {ts}",
        lambda c, ts: f"ERROR pvc-controller VolumeBindTimeout: PVC for {c['service']} unbound in {c['ns']} at {ts}",
        lambda c, ts: f"WARN ci-runner TestFailure: integration tests failed for {c['service']} in {c['ns']} at {ts}",
        lambda c, ts: f"ERROR traffic-manager SwitchFailed: blue/green aborted for {c['service']} in {c['ns']} at {ts}",
        lambda c, ts: f"ERROR istio-injector SidecarFailed: Envoy injection rejected for {c['service']} in {c['ns']} at {ts}",
        lambda c, ts: f"CRITICAL flyway MigrationFailed: schema {c['schema']} failed for {c['service']} at {ts}",
        lambda c, ts: f"ERROR argo-rollouts CanaryFailed: {c['canary_rate']}% errors for {c['service']} in {c['ns']} at {ts}",
    ],
    # Overlaps with network_issue (timeout/connection/unreachable) and deployment_issue (OOM/health check)
    "database_failure": [
        lambda c, ts: f"ERROR {c['service']} ConnectionRefused: cannot reach primary host at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} ConnectionPoolExhausted: max {c['pool_size']} connections reached at {ts}",
        lambda c, ts: f"ERROR {c['service']} QueryTimeoutException: query exceeded 30s for {c['client_svc']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} ReplicationLagCritical: lag={c['repl_lag']}s primary diverged at {ts}",
        lambda c, ts: f"ERROR {c['service']} DeadlockDetected: transaction rolled back for {c['client_svc']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} PrimaryUnavailable: failover in progress, replica not ready at {ts}",
        lambda c, ts: f"ERROR {c['service']} DiskIOSaturated: write latency {c['query_time']} for {c['client_svc']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} OOMKilled: memory limit exceeded, connections dropped at {ts}",
        lambda c, ts: f"WARN {c['service']} SlowQuery: blocking {c['pool_size']} connections for {c['query_time']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} HealthCheckFailed: {c['client_svc']} cannot connect to backend at {ts}",
    ],
    # Overlaps with authentication_failure (TLS/JWT/token/SSL/access)
    "certificate_expiry": [
        lambda c, ts: f"ERROR {c['service']} SSLCertExpiredException: certificate for {c['domain']} expired {c['days_expired']} days ago at {ts}",
        lambda c, ts: f"ERROR {c['service']} TLSHandshakeFailure: ssl verification failed for {c['domain']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} CertValidationError: JWT signing cert expired for {c['domain']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} X509CertExpired: presented cert not trusted by client for {c['domain']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} MTLSAuthFailed: client cert for {c['domain']} expired at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} SAMLCertExpired: assertion signing cert invalid for {c['domain']} at {ts}",
        lambda c, ts: f"ERROR {c['service']} TokenSigningFailed: RSA key cert expired, auth broken for {c['domain']} at {ts}",
        lambda c, ts: f"WARN {c['service']} CertExpiryWarning: {c['domain']} cert expires in {c['days_expired']} days at {ts}",
        lambda c, ts: f"ERROR {c['service']} AccessDenied: SSL certificate untrusted for user {c['user']} at {ts}",
        lambda c, ts: f"CRITICAL {c['service']} ChainValidationFailed: intermediate cert expired for {c['domain']} at {ts}",
    ],
}

ALERT_BUILDERS = {
    "network_issue": [
        lambda c: f"ALERT: packet_loss_rate on {c['service']} exceeded threshold 5% -> {_rand_pct()}% over 10min",
        lambda c: f"ALERT: dns_failures on {c['service']} exceeded 10/min -> {_rand_pct(11,80)}/min sustained",
        lambda c: f"ALERT: network_latency_ms on {c['service']} p99 exceeded 200ms -> {_rand_pct(201,900)}ms",
        lambda c: f"ALERT: firewall_drop_rate on {c['service']} exceeded 1% -> {_rand_pct(2,30)}% blocked",
        lambda c: f"ALERT: bgp_sessions on {c['service']} below threshold 2 -> {random.randint(0,1)} active",
        lambda c: f"ALERT: wan_throughput on {c['service']} dropped below 500Mbps -> {_rand_pct(10,490)}Mbps",
        lambda c: f"ALERT: vpn_stability on {c['service']} below 99% -> {_rand_pct(60,98)}% uptime",
        lambda c: f"ALERT: lb_5xx_rate on {c['service']} exceeded 1% -> {_rand_pct(2,40)}% upstream errors",
        lambda c: f"ALERT: nat_failures on {c['service']} exceeded 0.1% -> {_rand_pct(1,15)}% failing",
        lambda c: f"ALERT: dhcp_utilization on {c['service']} exceeded 90% -> {_rand_pct(91,99)}% pool used",
    ],
    "authentication_failure": [
        lambda c: f"ALERT: auth_error_rate on {c['service']} exceeded 5% -> {_rand_pct(6,40)}% over 10min",
        lambda c: f"ALERT: jwt_failures on {c['service']} exceeded 50/min -> {_rand_pct(51,200)}/min",
        lambda c: f"ALERT: sso_failure_rate on {c['service']} exceeded 10% -> {_rand_pct(11,60)}% failing",
        lambda c: f"ALERT: ldap_timeout_rate on {c['service']} exceeded 1% -> {_rand_pct(2,30)}% timing out",
        lambda c: f"ALERT: mfa_push_failures on {c['service']} exceeded 5% -> {_rand_pct(6,40)}% undelivered",
        lambda c: f"ALERT: api_401_rate on {c['service']} exceeded 2% -> {_rand_pct(3,30)}% unauthorized",
        lambda c: f"ALERT: session_expiry on {c['service']} exceeded 20/min -> {_rand_pct(21,100)}/min",
        lambda c: f"ALERT: kerberos_failures on {c['service']} exceeded 10/min -> {_rand_pct(11,60)}/min",
        lambda c: f"ALERT: oauth_latency_p95 on {c['service']} exceeded 500ms -> {_rand_pct(501,2000)}ms",
        lambda c: f"ALERT: ad_bind_failures on {c['service']} exceeded 5/min -> {_rand_pct(6,40)}/min",
    ],
    "deployment_issue": [
        lambda c: f"ALERT: pod_restarts on {c['service']} exceeded 3 -> {_rand_pct(4,20)} restarts in {c['ns']}",
        lambda c: f"ALERT: deployment_rollback triggered for {c['service']} in {c['ns']} — health check failing",
        lambda c: f"ALERT: oom_kills on {c['service']} exceeded 0 -> {_rand_pct(1,10)} kills in {c['ns']}",
        lambda c: f"ALERT: nodes_not_ready hosting {c['service']} -> {random.randint(1,3)} nodes in NotReady in {c['ns']}",
        lambda c: f"ALERT: pvc_unbound for {c['service']} in {c['ns']} -> {_rand_pct(61,300)}s waiting",
        lambda c: f"ALERT: ci_failure_rate for {c['service']} exceeded 20% -> {_rand_pct(21,80)}% failing",
        lambda c: f"ALERT: canary_error_rate on {c['service']} exceeded 5% -> {c['canary_rate']}% in {c['ns']}",
        lambda c: f"ALERT: image_pull_failures for {c['service']} in {c['ns']} -> {_rand_pct(1,10)} failures",
        lambda c: f"ALERT: migration_job {c['schema']} = failed for {c['service']} in {c['ns']}",
        lambda c: f"ALERT: sidecar_injection_failures for {c['service']} exceeded 0 -> {_rand_pct(1,5)} missing in {c['ns']}",
    ],
    # Overlaps with network_issue (connection/timeout) and deployment_issue (health_check/restart)
    "database_failure": [
        lambda c: f"ALERT: db_connection_failures on {c['service']} exceeded 1% -> {_rand_pct(2,30)}% failing",
        lambda c: f"ALERT: db_query_latency_p99 on {c['service']} exceeded 1s -> {c['query_time']} observed",
        lambda c: f"ALERT: replication_lag on {c['service']} exceeded 30s -> {c['repl_lag']}s behind primary",
        lambda c: f"ALERT: connection_pool_usage on {c['service']} exceeded 90% -> {c['pool_size']} connections active",
        lambda c: f"ALERT: db_deadlocks on {c['service']} exceeded 5/min -> {_rand_pct(6,30)}/min detected",
        lambda c: f"ALERT: primary_unavailable on {c['service']} -> automatic failover not completed",
        lambda c: f"ALERT: disk_io_wait on {c['service']} exceeded 80% -> {_rand_pct(81,99)}% utilisation",
        lambda c: f"ALERT: db_oom_kills on {c['service']} exceeded 0 -> {_rand_pct(1,5)} kills in last hour",
        lambda c: f"ALERT: slow_queries on {c['service']} exceeded 10/min -> {_rand_pct(11,50)}/min blocking",
        lambda c: f"ALERT: health_check_failures on {c['service']} -> {c['client_svc']} reporting DB unreachable",
    ],
    # Overlaps with authentication_failure (auth/login/token/ssl)
    "certificate_expiry": [
        lambda c: f"ALERT: ssl_cert_expiry_days for {c['service']} below threshold 7 -> {c['days_expired']} days until expiry",
        lambda c: f"ALERT: tls_handshake_failures on {c['service']} exceeded 1% -> {_rand_pct(2,30)}% failing for {c['domain']}",
        lambda c: f"ALERT: cert_validation_errors on {c['service']} exceeded 0 -> {_rand_pct(1,20)} errors for {c['domain']}",
        lambda c: f"ALERT: ssl_auth_failures on {c['service']} exceeded 5% -> {_rand_pct(6,40)}% users affected",
        lambda c: f"ALERT: x509_expired_cert detected on {c['service']} for domain {c['domain']}",
        lambda c: f"ALERT: jwt_signing_cert_expiry on {c['service']} -> token validation failing for {c['domain']}",
        lambda c: f"ALERT: mtls_auth_failures on {c['service']} exceeded 1% -> {_rand_pct(2,25)}% connections rejected",
        lambda c: f"ALERT: saml_cert_invalid on {c['service']} -> SSO login broken for {c['domain']}",
        lambda c: f"ALERT: access_denied_rate on {c['service']} exceeded 10% -> {_rand_pct(11,60)}% due to cert error",
        lambda c: f"ALERT: certificate_chain_broken on {c['service']} for {c['domain']} -> {_rand_pct(1,10)} services affected",
    ],
}

PRIORITY_MAP = {
    "network_issue":          {"ticket": "medium", "log": "medium", "alert": "high"},
    "authentication_failure": {"ticket": "high",   "log": "high",   "alert": "high"},
    "deployment_issue":       {"ticket": "medium", "log": "high",   "alert": "medium"},
    "database_failure":       {"ticket": "high",   "log": "high",   "alert": "high"},
    "certificate_expiry":     {"ticket": "high",   "log": "high",   "alert": "high"},
}

ALL_SOURCE_TYPES = ["ticket", "log", "alert"]


def _derive_priority(text: str, source_type: str, label: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in HIGH_KEYWORDS):
        return "high"
    if any(kw in text_lower for kw in MEDIUM_KEYWORDS):
        return "medium"
    return PRIORITY_MAP[label][source_type]


def _group_context(label: str) -> dict:
    if label == "network_issue":
        vlan2 = random.choice([10, 20, 30, 50, 100, 200])
        return {
            "service": random.choice(SERVICES_NETWORK),
            "src_ip": _rand_ip(),
            "dst_ip": _rand_ip(),
            "subnet": _rand_subnet(),
            "port": _rand_port(),
            "vlan": _rand_vlan(),
            "vlan2": vlan2,
        }
    if label == "authentication_failure":
        return {
            "service": random.choice(SERVICES_AUTH),
            "user": random.choice(USERS),
            "src_ip": _rand_ip(),
            "dst_ip": _rand_ip(),
        }
    if label == "deployment_issue":
        return {
            "service": random.choice(SERVICES_DEPLOY),
            "ns": random.choice(NAMESPACES),
            "schema": random.choice(SCHEMA_VERSIONS),
            "canary_rate": random.choice(CANARY_RATES),
        }
    if label == "database_failure":
        return {
            "service": random.choice(SERVICES_DB),
            "client_svc": random.choice(CLIENT_SERVICES),
            "query_time": random.choice(QUERY_TIMES),
            "repl_lag": random.randint(30, 300),
            "pool_size": random.choice([10, 20, 50, 100]),
        }
    return {
        "service": random.choice(SERVICES_CERT),
        "user": random.choice(USERS),
        "domain": random.choice(DOMAINS),
        "days_expired": random.randint(1, 10),
    }


def generate_incidents(
    n_groups_per_label: dict[str, int] | int = 50,
    source_coverage_probs: tuple[float, float, float] = (1.0, 0.0, 0.0),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    random.seed(random_state)

    base_labels = ["network_issue", "authentication_failure", "deployment_issue"]

    if isinstance(n_groups_per_label, int):
        label_counts = {label: n_groups_per_label for label in base_labels}
    else:
        label_counts = n_groups_per_label

    base_time = datetime(2024, 1, 15, 0, 0, 0)
    rows = []
    incident_id = 0
    group_id = 0

    for label, n_groups in label_counts.items():
        n_builders = len(TICKET_BUILDERS[label])

        for group_idx in range(n_groups):
            ctx = _group_context(label)
            tmpl_idx = group_idx % n_builders

            n_sources = random.choices([3, 2, 1], weights=source_coverage_probs, k=1)[0]
            selected_sources = random.sample(ALL_SOURCE_TYPES, n_sources)

            group_base_ts = base_time + timedelta(
                days=group_idx // 5,
                hours=(group_idx % 5) * 4,
                minutes=random.randint(0, 30),
            )
            log_ts = group_base_ts + timedelta(minutes=random.randint(1, 15))
            alert_ts = group_base_ts + timedelta(minutes=random.randint(5, 30))

            source_data = {
                "ticket": (TICKET_BUILDERS[label][tmpl_idx](ctx), group_base_ts),
                "log": (LOG_BUILDERS[label][tmpl_idx](ctx, log_ts.strftime("%Y-%m-%d %H:%M:%S")), log_ts),
                "alert": (ALERT_BUILDERS[label][tmpl_idx](ctx), alert_ts),
            }

            for source_type in selected_sources:
                text, ts = source_data[source_type]
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
    parser = argparse.ArgumentParser(description="Generate synthetic incident dataset")
    parser.add_argument(
        "--realistic",
        action="store_true",
        help=(
            "Generate a realistic imbalanced dataset with 5 categories and variable "
            "group completeness (1–3 sources per group). Saved to synthetic_incidents_realistic.csv."
        ),
    )
    args = parser.parse_args()

    if args.realistic:
        output_path = REALISTIC_OUTPUT_PATH
        label_counts = {
            "authentication_failure": 100,
            "network_issue": 70,
            "deployment_issue": 55,
            "database_failure": 35,
            "certificate_expiry": 20,
        }
        coverage_probs = (0.60, 0.25, 0.15)
        df = generate_incidents(
            n_groups_per_label=label_counts,
            source_coverage_probs=coverage_probs,
        )
    else:
        output_path = OUTPUT_PATH
        df = generate_incidents()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows -> {output_path}")
    print(df["label"].value_counts().to_string())
    print(df["priority"].value_counts().to_string())
    print(df["source_type"].value_counts().to_string())
