"""Generate synthetic Software Development Analytics sample data.

Creates all JSON files required by the ingestion pipeline.

Run once before ingestion:
    python -m data.generate_sample
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

SAMPLE_DIR = Path(__file__).parent / "sample"
SAMPLE_DIR.mkdir(exist_ok=True)

# ── Volume targets ────────────────────────────────────────────────────────────
N_TEAMS = 12
N_DEVS = 80
N_COMPONENTS = 70
N_REPOS = 35
N_EPICS = 40
N_DEPENDS_ON = 90
N_TICKETS = 2500
N_SPRINTS_PER_TEAM = 20
N_DEPLOYMENTS = 4000
N_TEST_COV = 700
N_INCIDENTS = 300
N_ARCH_DOCS = 50
N_POSTMORTEMS = 40

# Anomaly targets — pick real component IDs after generation
ANOMALY_COMP_INDICES = [4, 11, 30]  # 0-based → comp-005, comp-012, comp-031

DEPARTMENTS = ["backend", "frontend", "platform", "data", "security"]
SENIORITY_LEVELS = (
    ["junior"] * 24 + ["mid"] * 32 + ["senior"] * 16 + ["lead"] * 8
)
COMPONENT_TYPES = (
    ["service"] * 28 + ["library"] * 18 + ["database"] * 14
    + ["frontend"] * 7 + ["gateway"] * 3
)
LANGUAGES = ["Python", "Go", "TypeScript", "Java", "Rust", "Kotlin", "C++"]
TICKET_TYPES = ["bug"] * 500 + ["feature"] * 1250 + ["tech_debt"] * 500 + ["incident"] * 250
PRIORITIES = ["P1", "P2", "P3", "P4"]
TICKET_STATUSES = ["open", "in_progress", "done", "closed"]
DEPLOYMENT_STATUSES = ["success"] * 70 + ["failed"] * 15 + ["rollback"] * 15
INCIDENT_SEVERITIES = ["P1"] * 30 + ["P2"] * 90 + ["P3"] * 180

# ── Text templates ────────────────────────────────────────────────────────────
BUG_TEMPLATES = [
    "{component} becomes unresponsive when {condition}",
    "{component} throws {error} under {load}",
    "{component} fails intermittently during {operation}",
    "Memory leak detected in {component} after {duration} of continuous operation",
    "{component} returns incorrect results when {condition}",
    "Race condition in {component} causes data corruption under concurrent {load}",
    "{component} crashes with {error} on startup in {env}",
    "Timeout errors in {component} spike above threshold during {operation}",
    "{component} does not handle {edge_case} gracefully",
    "{component} logs {error} repeatedly when processing large payloads",
]
CONDITIONS = [
    "concurrent connections exceed 500",
    "the database connection pool is exhausted",
    "a downstream service returns 503",
    "memory usage exceeds 80%",
    "the event queue is full",
    "requests arrive faster than the rate limit allows",
]
ERRORS = [
    "NullPointerException", "ConnectionTimeoutError", "OutOfMemoryError",
    "IndexOutOfBoundsException", "DeadlockException", "ConcurrentModificationException",
]
LOADS = [
    "high traffic", "peak load", "sustained load above 1000 RPS",
    "burst traffic", "concurrent writes",
]
OPERATIONS = [
    "database migrations", "cache invalidation", "batch processing",
    "report generation", "authentication flow", "API sync",
]
EDGE_CASES = [
    "empty payloads", "null values in required fields",
    "unicode characters in user input", "very large request bodies",
]
ENVS = ["production", "staging", "canary environment"]

FEATURE_TEMPLATES = [
    "Add {feature} to {component} to support {use_case}",
    "Implement {feature} endpoint in {component}",
    "Extend {component} with {feature} capability for {use_case}",
    "Build {feature} integration layer between {component} and downstream services",
    "Add pagination support to {component} {feature} API",
    "Introduce {feature} caching layer in {component} to reduce latency",
]
FEATURES = [
    "bulk export", "real-time streaming", "role-based access control",
    "audit logging", "multi-tenancy", "webhook", "GraphQL",
    "async job queue", "rate limiting", "circuit breaker",
]
USE_CASES = [
    "enterprise customers", "compliance reporting", "mobile clients",
    "third-party integrations", "SLA monitoring",
]

TECH_DEBT_TEMPLATES = [
    "Refactor {component} to remove deprecated {artifact}",
    "Migrate {component} from {old_tech} to {new_tech}",
    "Extract {artifact} from {component} into a shared library",
    "Remove hardcoded configuration from {component}",
    "Replace synchronous calls in {component} with async equivalents",
    "Update {component} dependencies to address known vulnerabilities",
]
ARTIFACTS = ["global state", "singleton pattern", "monolithic handler", "raw SQL queries"]
OLD_TECHS = ["Python 2 code", "jQuery", "REST polling", "XML parsing"]
NEW_TECHS = ["async/await", "GraphQL subscriptions", "event-driven messaging", "Protobuf"]

ADR_TITLES = [
    "Use PostgreSQL as the primary relational store",
    "Adopt event-driven architecture for inter-service communication",
    "Implement circuit breaker pattern for external API calls",
    "Use Redis for distributed caching and session storage",
    "Adopt OpenTelemetry for observability across all services",
    "Migrate authentication to OAuth 2.0 / OIDC",
    "Use Kafka for high-throughput event streaming",
    "Adopt hexagonal architecture for domain isolation",
    "Implement CQRS for read/write separation in reporting services",
    "Use Kubernetes for container orchestration",
    "Adopt blue-green deployment strategy for zero-downtime releases",
    "Use gRPC for internal service communication",
    "Implement distributed tracing with Jaeger",
    "Adopt API gateway pattern for ingress traffic management",
    "Use infrastructure-as-code with Terraform",
    "Implement feature flags with LaunchDarkly",
    "Adopt trunk-based development as branching strategy",
    "Use columnstore indexes for analytical queries",
    "Implement rate limiting at the API gateway layer",
    "Adopt chaos engineering practices for resilience testing",
    "Migrate from monolith to microservices for the billing domain",
    "Use vector database for semantic search capabilities",
    "Implement data mesh architecture for domain ownership",
    "Adopt zero-trust network security model",
    "Use Rust for performance-critical path processing",
    "Implement multi-region active-active replication",
    "Adopt GraphQL federation for unified API layer",
    "Use Flink for real-time stream processing",
    "Implement saga pattern for distributed transactions",
    "Adopt eBPF for low-overhead kernel-level observability",
]
ADR_STATUSES = ["Accepted"] * 18 + ["Proposed"] * 8 + ["Deprecated"] * 4

POSTMORTEM_ROOT_CAUSES = [
    "upstream service returned 503 for an extended period",
    "third-party vendor experienced an outage affecting our data pipeline",
    "external API rate limit was exceeded due to a misconfigured retry loop",
    "cascading failure from a dependent component overwhelmed the message queue",
    "configuration drift introduced during deployment caused an incompatible schema version",
    "a memory leak in the worker process caused gradual OOM terminations",
    "an expired TLS certificate was not rotated before the deadline",
    "a database index was accidentally dropped during a migration",
    "autoscaling policy was misconfigured and failed to provision capacity",
    "a dependency upgrade introduced a breaking change in the serialization format",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fid(prefix: str, n: int, total: int) -> str:
    width = max(3, len(str(total)))
    return f"{prefix}-{str(n).zfill(width)}"


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _rand_dt(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


# ── Generators ────────────────────────────────────────────────────────────────

def _gen_teams() -> list[dict]:
    teams = []
    dept_cycle = (DEPARTMENTS * 3)[:N_TEAMS]
    random.shuffle(dept_cycle)
    for i in range(1, N_TEAMS + 1):
        teams.append({
            "id": _fid("team", i, N_TEAMS),
            "name": f"{fake.word().capitalize()} {dept_cycle[i-1].capitalize()} Team",
            "department": dept_cycle[i - 1],
        })
    return teams


def _gen_repositories() -> list[dict]:
    repos = []
    for i in range(1, N_REPOS + 1):
        lang = random.choice(LANGUAGES)
        name = f"{fake.word()}-{fake.word()}".lower()
        repos.append({
            "id": _fid("repo", i, N_REPOS),
            "name": name,
            "url": f"https://github.com/acme-corp/{name}",
            "language": lang,
        })
    return repos


def _gen_developers(teams: list[dict]) -> list[dict]:
    devs = []
    seniority_pool = SENIORITY_LEVELS[:]
    random.shuffle(seniority_pool)
    team_ids = [t["id"] for t in teams]
    for i in range(1, N_DEVS + 1):
        devs.append({
            "id": _fid("dev", i, N_DEVS),
            "name": fake.name(),
            "email": fake.email(),
            "seniority": seniority_pool[i - 1],
            "team_id": random.choice(team_ids),
        })
    return devs


def _gen_components(teams: list[dict], repos: list[dict], devs: list[dict]) -> list[dict]:
    comps = []
    type_pool = COMPONENT_TYPES[:]
    random.shuffle(type_pool)
    repo_ids = [r["id"] for r in repos]
    team_ids = [t["id"] for t in teams]
    dev_ids = [d["id"] for d in devs]
    for i in range(1, N_COMPONENTS + 1):
        comps.append({
            "id": _fid("comp", i, N_COMPONENTS),
            "name": f"{fake.word()}-{type_pool[i-1]}".lower(),
            "type": type_pool[i - 1],
            "team_id": random.choice(team_ids),
            "owner_id": random.choice(dev_ids),
            "repo_id": random.choice(repo_ids),
        })
    return comps


def _gen_epics(teams: list[dict], comps: list[dict]) -> list[dict]:
    epics = []
    team_ids = [t["id"] for t in teams]
    comp_ids = [c["id"] for c in comps]
    statuses = ["active"] * 20 + ["completed"] * 15 + ["on-hold"] * 5
    random.shuffle(statuses)
    for i in range(1, N_EPICS + 1):
        n_comps = random.randint(2, 6)
        epics.append({
            "id": _fid("epic", i, N_EPICS),
            "title": f"{fake.bs().capitalize()}",
            "status": statuses[i - 1],
            "team_id": random.choice(team_ids),
            "component_ids": random.sample(comp_ids, n_comps),
        })
    return epics


def _gen_depends_on(comps: list[dict], anomaly_comp_ids: list[str]) -> list[dict]:
    """Generate 90 DEPENDS_ON edges with depth-3+ chains and high-fan-in nodes."""
    comp_ids = [c["id"] for c in comps]
    edges: set[tuple[str, str]] = set()
    edge_list: list[dict] = []

    def add_edge(src: str, dst: str, etype: str = "hard") -> None:
        if src != dst and (src, dst) not in edges:
            edges.add((src, dst))
            edge_list.append({"from_id": src, "to_id": dst, "type": etype})

    # Force 3 explicit chains of depth ≥ 3
    chains = [
        comp_ids[0:4],    # depth 3 chain
        comp_ids[10:14],  # depth 3 chain
        comp_ids[20:24],  # depth 3 chain
    ]
    for chain in chains:
        for k in range(len(chain) - 1):
            add_edge(chain[k], chain[k + 1])

    # Force 30% of components to have ≥ 2 incoming edges
    high_fan_in = random.sample(comp_ids, N_COMPONENTS // 3)
    for target in high_fan_in:
        sources = [c for c in comp_ids if c != target]
        for src in random.sample(sources, 2):
            add_edge(src, target, "soft")

    # Fill remaining edges randomly (avoid cycles greedily)
    attempts = 0
    while len(edge_list) < N_DEPENDS_ON and attempts < 5000:
        src, dst = random.sample(comp_ids, 2)
        add_edge(src, dst, random.choice(["hard", "soft"]))
        attempts += 1

    return edge_list[:N_DEPENDS_ON]


def _gen_member_of(devs: list[dict]) -> list[dict]:
    rels = []
    start = datetime(2019, 1, 1)
    end = datetime(2023, 1, 1)
    for dev in devs:
        rels.append({
            "developer_id": dev["id"],
            "team_id": dev["team_id"],
            "since": _date(_rand_dt(start, end)),
        })
    return rels


def _gen_contributed_to(devs: list[dict], comps: list[dict]) -> list[dict]:
    rels = []
    comp_ids = [c["id"] for c in comps]
    for dev in devs:
        n = random.randint(1, 6)
        for comp_id in random.sample(comp_ids, n):
            rels.append({
                "developer_id": dev["id"],
                "component_id": comp_id,
                "commits": random.randint(1, 200),
            })
    return rels


def _ticket_description(ticket_type: str, comp_name: str) -> str:
    if ticket_type == "bug":
        tmpl = random.choice(BUG_TEMPLATES)
        return tmpl.format(
            component=comp_name,
            condition=random.choice(CONDITIONS),
            error=random.choice(ERRORS),
            load=random.choice(LOADS),
            operation=random.choice(OPERATIONS),
            duration=f"{random.randint(1, 72)} hours",
            edge_case=random.choice(EDGE_CASES),
            env=random.choice(ENVS),
        )
    if ticket_type == "feature":
        tmpl = random.choice(FEATURE_TEMPLATES)
        return tmpl.format(
            component=comp_name,
            feature=random.choice(FEATURES),
            use_case=random.choice(USE_CASES),
        )
    if ticket_type == "tech_debt":
        tmpl = random.choice(TECH_DEBT_TEMPLATES)
        return tmpl.format(
            component=comp_name,
            artifact=random.choice(ARTIFACTS),
            old_tech=random.choice(OLD_TECHS),
            new_tech=random.choice(NEW_TECHS),
        )
    # incident type
    return (
        f"Incident in {comp_name}: {random.choice(CONDITIONS).capitalize()}. "
        f"Immediate mitigation applied. Root cause investigation in progress."
    )


def _gen_tickets(comps: list[dict], devs: list[dict], teams: list[dict],
                 anomaly_comp_ids: list[str]) -> list[dict]:
    tickets = []
    type_pool = TICKET_TYPES[:]
    random.shuffle(type_pool)
    comp_by_id = {c["id"]: c for c in comps}
    dev_ids = [d["id"] for d in devs]
    team_ids = [t["id"] for t in teams]
    sprint_ids = [_fid("sprint", i, N_SPRINTS_PER_TEAM) for i in range(1, N_SPRINTS_PER_TEAM + 1)]
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    for i in range(N_TICKETS):
        ttype = type_pool[i % len(type_pool)]
        comp = random.choice(comps)
        created = _rand_dt(start, end)
        status = random.choice(TICKET_STATUSES)
        resolved = None
        if status in ("done", "closed"):
            resolved = _ts(created + timedelta(days=random.randint(1, 30)))

        # Normal reopen count
        reopen = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]

        priority = random.choices(PRIORITIES, weights=[5, 30, 45, 20])[0]
        tickets.append({
            "id": str(uuid.uuid4()),
            "title": _ticket_description(ttype, comp["name"]).split(".")[0][:120],
            "type": ttype,
            "priority": priority,
            "status": status,
            "component_id": comp["id"],
            "assignee_id": random.choice(dev_ids),
            "team_id": comp.get("team_id", random.choice(team_ids)),
            "story_points": random.choice([1, 2, 3, 5, 8, 13]),
            "created_at": _ts(created),
            "resolved_at": resolved,
            "reopened_count": reopen,
            "sprint_id": random.choice(sprint_ids),
            "description": _ticket_description(ttype, comp["name"])
            + " " + fake.paragraph(nb_sentences=3),
        })

    # Anomaly 1: bug reopen hotspots — force avg reopened_count > 3 for 3 components
    for comp_id in anomaly_comp_ids:
        affected = [t for t in tickets if t["component_id"] == comp_id and t["type"] == "bug"]
        for t in affected[:max(5, len(affected))]:
            t["reopened_count"] = random.randint(4, 8)

    return tickets


def _gen_sprint_metrics(teams: list[dict], anomaly_team_ids: list[str]) -> list[dict]:
    metrics = []
    sprint_start = datetime(2023, 1, 2)
    sprint_len = timedelta(days=37)  # ~20 sprints over 2 years

    for team in teams:
        for s in range(1, N_SPRINTS_PER_TEAM + 1):
            s_start = sprint_start + sprint_len * (s - 1)
            s_end = s_start + timedelta(days=13)
            planned = random.randint(30, 60)
            # Normal velocity 0.75 – 1.0
            if team["id"] in anomaly_team_ids and s >= N_SPRINTS_PER_TEAM - 4:
                # Anomaly 2: velocity decline — last 5 sprints < 0.65
                velocity = round(random.uniform(0.45, 0.64), 2)
            else:
                velocity = round(random.uniform(0.70, 1.05), 2)
            completed = int(planned * velocity)
            metrics.append({
                "id": f"sm-{team['id']}-sprint-{s:02d}",
                "sprint_id": _fid("sprint", s, N_SPRINTS_PER_TEAM),
                "team_id": team["id"],
                "start_date": _date(s_start),
                "end_date": _date(s_end),
                "planned_points": planned,
                "completed_points": completed,
                "velocity": velocity,
                "bug_count": random.randint(0, 8),
                "feature_count": random.randint(2, 12),
                "carried_over_count": random.randint(0, 5),
            })

    return metrics


def _gen_deployments(comps: list[dict], devs: list[dict],
                     anomaly_comp_ids: list[str]) -> list[dict]:
    deployments = []
    dev_ids = [d["id"] for d in devs]
    comp_ids = [c["id"] for c in comps]
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    status_pool = DEPLOYMENT_STATUSES[:]

    for _ in range(N_DEPLOYMENTS):
        comp_id = random.choice(comp_ids)
        status = random.choice(status_pool)
        deployed_at = _rand_dt(start, end)
        deployments.append({
            "id": str(uuid.uuid4()),
            "component_id": comp_id,
            "version": f"{random.randint(1,5)}.{random.randint(0,20)}.{random.randint(0,50)}",
            "status": status,
            "environment": random.choices(["production", "staging"], weights=[60, 40])[0],
            "deployed_by": random.choice(dev_ids),
            "deployed_at": _ts(deployed_at),
            "duration_seconds": random.randint(30, 600),
        })

    # Anomaly 3: rollback hotspots — force >20% rollback rate for 3 components
    for comp_id in anomaly_comp_ids:
        comp_deps = [d for d in deployments if d["component_id"] == comp_id]
        rollback_target = max(int(len(comp_deps) * 0.25), 5)
        for dep in comp_deps[:rollback_target]:
            dep["status"] = "rollback"

    return deployments


def _gen_test_coverage(comps: list[dict], anomaly_comp_ids: list[str]) -> list[dict]:
    records = []
    for comp in comps:
        base_cov = round(random.uniform(55.0, 92.0), 1)
        base_tests = random.randint(50, 800)
        meas_start = datetime(2024, 1, 1)
        for m in range(10):
            meas_date = meas_start + timedelta(days=37 * m)
            noise = round(random.uniform(-3.0, 3.0), 1)
            line_cov = max(10.0, min(99.9, base_cov + noise))
            records.append({
                "id": str(uuid.uuid4()),
                "component_id": comp["id"],
                "measured_at": _date(meas_date),
                "line_coverage": line_cov,
                "branch_coverage": round(line_cov * random.uniform(0.80, 0.95), 1),
                "test_count": base_tests + random.randint(-20, 40),
            })

    # Anomaly 4: coverage decline — 4 consecutive declining measurements
    for comp_id in anomaly_comp_ids:
        comp_recs = sorted(
            [r for r in records if r["component_id"] == comp_id],
            key=lambda r: r["measured_at"],
        )
        if len(comp_recs) >= 4:
            start_cov = 82.0
            for k, rec in enumerate(comp_recs[-4:]):
                rec["line_coverage"] = round(start_cov - k * 7.0, 1)
                rec["branch_coverage"] = round(rec["line_coverage"] * 0.88, 1)

    return records


def _gen_incidents(comps: list[dict], anomaly_comp_ids: list[str]) -> list[dict]:
    incidents = []
    comp_ids = [c["id"] for c in comps]
    severity_pool = INCIDENT_SEVERITIES[:]
    random.shuffle(severity_pool)
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    for i in range(N_INCIDENTS):
        sev = severity_pool[i % len(severity_pool)]
        comp_id = random.choice(comp_ids)
        started = _rand_dt(start, end)
        duration = random.randint(15, 480)
        resolved = started + timedelta(minutes=duration)
        incidents.append({
            "id": str(uuid.uuid4()),
            "title": f"{sev} incident: {fake.bs()[:80]}",
            "severity": sev,
            "component_id": comp_id,
            "root_cause_component_id": None,
            "started_at": _ts(started),
            "resolved_at": _ts(resolved),
            "duration_minutes": duration,
            "affected_users": random.randint(0, 50000),
            "status": "resolved",
        })

    # Anomaly 5: 6 P1 incidents in Q1 2024 with root_cause_component_id
    q1_start = datetime(2024, 1, 1)
    q1_end = datetime(2024, 3, 31)
    for k in range(6):
        comp_id = anomaly_comp_ids[k % len(anomaly_comp_ids)]
        root_cause = random.choice([c for c in comp_ids if c != comp_id])
        started = _rand_dt(q1_start, q1_end)
        duration = random.randint(60, 300)
        incidents.append({
            "id": str(uuid.uuid4()),
            "title": f"P1 incident: {fake.bs()[:80]}",
            "severity": "P1",
            "component_id": comp_id,
            "root_cause_component_id": root_cause,
            "started_at": _ts(started),
            "resolved_at": _ts(started + timedelta(minutes=duration)),
            "duration_minutes": duration,
            "affected_users": random.randint(5000, 100000),
            "status": "resolved",
        })

    return incidents


def _adr_text(n: int, title: str, status: str) -> str:
    ctx_p1 = fake.paragraph(nb_sentences=4)
    ctx_p2 = fake.paragraph(nb_sentences=3)
    decision = fake.paragraph(nb_sentences=3)
    # Ensure ≥10 ADRs mention "critical" / "high risk" / "architecturally sensitive"
    risk_phrase = ""
    if n <= 10:
        risk_phrase = random.choice([
            " This decision is **critical** to system reliability.",
            " The team considers this **high risk** without a mitigation plan.",
            " This area is **architecturally sensitive** and requires careful review.",
        ])
    return (
        f"# ADR-{n:03d}: {title}\n\n"
        f"## Status: {status}\n\n"
        f"## Context\n\n{ctx_p1}{risk_phrase}\n\n{ctx_p2}\n\n"
        f"## Decision\n\n{decision}\n\n"
        f"## Consequences\n\n"
        f"- {fake.sentence()}\n"
        f"- {fake.sentence()}\n"
        f"- {fake.sentence()}\n"
    )


def _design_doc_text(title: str) -> str:
    return (
        f"# Design Document: {title}\n\n"
        f"## Overview\n\n{fake.paragraph(nb_sentences=4)}\n\n"
        f"## Goals\n\n- {fake.sentence()}\n- {fake.sentence()}\n- {fake.sentence()}\n\n"
        f"## Non-Goals\n\n- {fake.sentence()}\n\n"
        f"## Proposed Design\n\n{fake.paragraph(nb_sentences=5)}\n\n"
        f"## Alternatives Considered\n\n{fake.paragraph(nb_sentences=3)}\n\n"
        f"## Open Questions\n\n- {fake.sentence()}\n"
    )


def _rfc_text(title: str) -> str:
    return (
        f"# RFC: {title}\n\n"
        f"## Motivation\n\n{fake.paragraph(nb_sentences=3)}\n\n"
        f"## Proposal\n\n{fake.paragraph(nb_sentences=4)}\n\n"
        f"## Drawbacks\n\n{fake.paragraph(nb_sentences=2)}\n\n"
        f"## Unresolved Questions\n\n- {fake.sentence()}\n- {fake.sentence()}\n"
    )


def _gen_arch_docs(comps: list[dict]) -> list[dict]:
    docs = []
    comp_ids = [c["id"] for c in comps]
    # adr 60% (30), design_doc 30% (15), rfc 10% (5)
    doc_types = ["adr"] * 30 + ["design_doc"] * 15 + ["rfc"] * 5
    random.shuffle(doc_types)
    titles = random.sample(ADR_TITLES * 2, N_ARCH_DOCS)
    statuses = (ADR_STATUSES * 2)[:N_ARCH_DOCS]

    for i in range(1, N_ARCH_DOCS + 1):
        dtype = doc_types[i - 1]
        title = titles[i - 1]
        status = statuses[i - 1]
        date = _date(_rand_dt(datetime(2021, 1, 1), datetime(2024, 6, 1)))
        if dtype == "adr":
            text = _adr_text(i, title, status)
        elif dtype == "design_doc":
            text = _design_doc_text(title)
        else:
            text = _rfc_text(title)
        docs.append({
            "id": f"adr-{i:03d}",
            "component_id": random.choice(comp_ids),
            "doc_type": dtype,
            "title": title,
            "date": date,
            "text": text,
        })
    return docs


def _postmortem_text(title: str, severity: str, root_cause: str) -> str:
    summary = fake.paragraph(nb_sentences=3)
    timeline = (
        f"- {fake.time_object().strftime('%H:%M')} — Alert fired: {fake.bs()}\n"
        f"- {fake.time_object().strftime('%H:%M')} — On-call engineer paged\n"
        f"- {fake.time_object().strftime('%H:%M')} — Root cause identified\n"
        f"- {fake.time_object().strftime('%H:%M')} — Mitigation applied\n"
        f"- {fake.time_object().strftime('%H:%M')} — Service restored\n"
    )
    action_items = (
        f"- [ ] {fake.sentence()}\n"
        f"- [ ] {fake.sentence()}\n"
        f"- [ ] {fake.sentence()}\n"
    )
    return (
        f"# Postmortem: {title}\n\n"
        f"**Severity**: {severity}\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## Root Cause\n\nThe root cause was determined to be: {root_cause}.\n\n"
        f"## Timeline\n\n{timeline}\n"
        f"## Impact\n\n{fake.paragraph(nb_sentences=2)}\n\n"
        f"## Action Items\n\n{action_items}"
    )


def _gen_postmortems(comps: list[dict], incidents: list[dict]) -> list[dict]:
    postmortems = []
    comp_ids = [c["id"] for c in comps]
    inc_ids = [inc["id"] for inc in incidents if inc["severity"] in ("P1", "P2")]
    severities = ["P1"] * 8 + ["P2"] * 20 + ["P3"] * 12

    for i in range(1, N_POSTMORTEMS + 1):
        sev = severities[(i - 1) % len(severities)]
        root_cause = random.choice(POSTMORTEM_ROOT_CAUSES)
        title = fake.bs()[:80]
        date = _date(_rand_dt(datetime(2023, 1, 1), datetime(2024, 12, 31)))
        postmortems.append({
            "id": f"pm-{i:03d}",
            "incident_id": random.choice(inc_ids) if inc_ids else str(uuid.uuid4()),
            "component_id": random.choice(comp_ids),
            "severity": sev,
            "title": title,
            "date": date,
            "text": _postmortem_text(title, sev, root_cause),
        })
    return postmortems


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    log.info("Generating shared_ids …")
    teams = _gen_teams()
    repos = _gen_repositories()
    devs = _gen_developers(teams)
    comps = _gen_components(teams, repos, devs)
    epics = _gen_epics(teams, comps)

    anomaly_comp_ids = [comps[i]["id"] for i in ANOMALY_COMP_INDICES]
    anomaly_team_ids = [teams[i]["id"] for i in [0, 3, 7]]

    shared = {
        "teams": teams,
        "developers": devs,
        "components": comps,
        "repositories": repos,
        "epics": epics,
    }
    (SAMPLE_DIR / "shared_ids.json").write_text(json.dumps(shared, indent=2))
    log.info("  teams=%d devs=%d comps=%d repos=%d epics=%d",
             len(teams), len(devs), len(comps), len(repos), len(epics))

    log.info("Generating graph_relationships …")
    rels = {
        "member_of": _gen_member_of(devs),
        "depends_on": _gen_depends_on(comps, anomaly_comp_ids),
        "contributed_to": _gen_contributed_to(devs, comps),
    }
    (SAMPLE_DIR / "graph_relationships.json").write_text(json.dumps(rels, indent=2))
    log.info("  member_of=%d depends_on=%d contributed_to=%d",
             len(rels["member_of"]), len(rels["depends_on"]), len(rels["contributed_to"]))

    log.info("Generating tickets …")
    tickets = _gen_tickets(comps, devs, teams, anomaly_comp_ids)
    (SAMPLE_DIR / "tickets.json").write_text(json.dumps(tickets, indent=2))
    log.info("  tickets=%d", len(tickets))

    log.info("Generating sprint_metrics …")
    sprint_metrics = _gen_sprint_metrics(teams, anomaly_team_ids)
    (SAMPLE_DIR / "sprint_metrics.json").write_text(json.dumps(sprint_metrics, indent=2))
    log.info("  sprint_metrics=%d", len(sprint_metrics))

    log.info("Generating deployments …")
    deployments = _gen_deployments(comps, devs, anomaly_comp_ids)
    (SAMPLE_DIR / "deployments.json").write_text(json.dumps(deployments, indent=2))
    log.info("  deployments=%d", len(deployments))

    log.info("Generating test_coverage …")
    test_coverage = _gen_test_coverage(comps, anomaly_comp_ids)
    (SAMPLE_DIR / "test_coverage.json").write_text(json.dumps(test_coverage, indent=2))
    log.info("  test_coverage=%d", len(test_coverage))

    log.info("Generating incidents …")
    incidents = _gen_incidents(comps, anomaly_comp_ids)
    (SAMPLE_DIR / "incidents.json").write_text(json.dumps(incidents, indent=2))
    log.info("  incidents=%d", len(incidents))

    log.info("Generating arch_docs …")
    arch_docs = _gen_arch_docs(comps)
    (SAMPLE_DIR / "arch_docs.json").write_text(json.dumps(arch_docs, indent=2))
    log.info("  arch_docs=%d", len(arch_docs))

    log.info("Generating postmortems …")
    postmortems = _gen_postmortems(comps, incidents)
    (SAMPLE_DIR / "postmortems.json").write_text(json.dumps(postmortems, indent=2))
    log.info("  postmortems=%d", len(postmortems))

    log.info("Done — all sample files written to %s", SAMPLE_DIR)


if __name__ == "__main__":
    main()
