from agents.answer_agent import _format_results_for_prompt
from agents.judge_agent import CriteriaScores, JudgeDecision


def make_judge_decision(decision: str, gaps=None, conflicts=None):
    return JudgeDecision(
        decision=decision,
        criteria_scores=CriteriaScores(completeness=4, relevance=4, consistency=4, specificity=4),
        gaps=gaps or [],
        conflicts=conflicts or [],
        reformulated_query=None,
        reasoning="test",
    )


def test_vector_result_has_header(sample_retrieval_results):
    output = _format_results_for_prompt([sample_retrieval_results[0]], None)
    assert "=== VECTOR AGENT RESULTS ===" in output


def test_graph_result_has_header(sample_retrieval_results):
    output = _format_results_for_prompt([sample_retrieval_results[1]], None)
    assert "=== GRAPH AGENT RESULTS ===" in output


def test_sql_result_has_header(sample_retrieval_results):
    output = _format_results_for_prompt([sample_retrieval_results[2]], None)
    assert "=== SQL AGENT RESULTS ===" in output


def test_conflicts_section_present(sample_retrieval_results):
    decision = make_judge_decision("ACCEPT", conflicts=["Vector states X, SQL states Y"])
    output = _format_results_for_prompt(sample_retrieval_results, decision)
    assert "=== JUDGE AGENT: CONFLICTS DETECTED ===" in output


def test_conflicts_section_absent_when_empty(sample_retrieval_results):
    decision = make_judge_decision("ACCEPT", conflicts=[])
    output = _format_results_for_prompt(sample_retrieval_results, decision)
    assert "CONFLICTS DETECTED" not in output


def test_gaps_section_on_max_iterations(sample_retrieval_results):
    decision = make_judge_decision("MAX_ITERATIONS_REACHED", gaps=["Missing graph data"])
    output = _format_results_for_prompt(sample_retrieval_results, decision)
    assert "=== JUDGE AGENT: UNRESOLVED GAPS" in output


def test_gaps_section_absent_on_accept(sample_retrieval_results):
    decision = make_judge_decision("ACCEPT", gaps=["some gap"])
    output = _format_results_for_prompt(sample_retrieval_results, decision)
    assert "UNRESOLVED GAPS" not in output


def test_none_judge_decision(sample_retrieval_results):
    output = _format_results_for_prompt(sample_retrieval_results, None)
    assert "CONFLICTS" not in output
    assert "UNRESOLVED GAPS" not in output
    assert isinstance(output, str)
