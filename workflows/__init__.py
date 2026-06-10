"""Workflows package.

Two orchestration variants:

- ``v1_workflow.run`` — Hierarchical without closed-loop.
  Orchestrator → parallel retrieval → Answer Agent.

- ``v2_workflow.run`` — Hierarchical with closed-loop.
  Retrieval loop (Orchestrator → parallel retrieval → Judge) → Answer Agent.
  Max 3 iterations enforced; Judge documents remaining gaps on forced acceptance.
"""

from workflows.registry import AGENT_REGISTRY  # noqa: F401  re-export for backwards compatibility
