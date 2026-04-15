from app.scope_index import query_scope_signal


def test_scope_signal_in_scope_policy_query() -> None:
    sig = query_scope_signal("What is the VPN policy for remote work?")
    assert sig["in_scope"] is True
    assert sig["match_ratio"] > 0


def test_scope_signal_out_of_scope_query() -> None:
    sig = query_scope_signal("What is corn and how does photosynthesis work?")
    assert sig["in_scope"] is False
    assert "corn" in sig["tokens"]
