import operator

from workflows.state import _merge_latencies


def test_merge_empty_dicts():
    assert _merge_latencies({}, {}) == {}


def test_merge_disjoint():
    assert _merge_latencies({"a": 1.0}, {"b": 2.0}) == {"a": 1.0, "b": 2.0}


def test_merge_overlapping_keys():
    assert _merge_latencies({"a": 1.0}, {"a": 2.0}) == {"a": 3.0}


def test_merge_does_not_mutate_input():
    a = {"x": 1.0}
    b = {"x": 0.5}
    _merge_latencies(a, b)
    assert a == {"x": 1.0}


def test_retrieval_results_reducer():
    list_a = [{"source": "vector"}]
    list_b = [{"source": "graph"}]
    result = operator.add(list_a, list_b)
    assert result == [{"source": "vector"}, {"source": "graph"}]
    assert list_a == [{"source": "vector"}]
