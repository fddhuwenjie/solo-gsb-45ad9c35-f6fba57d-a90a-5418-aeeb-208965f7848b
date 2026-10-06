"""End-to-end tests for JSONPath multi-location session replacements.

``POST /sessions/{id}/replacements`` evaluates one restricted JSONPath
(the ``/query`` grammar) on the current merged document, turns the at-most-20
pre-ordered matches into one actor's contiguous causal ``set`` operations,
and commits them with the ordinary merge rules in one atomic revision.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sessions import RevisionConflictError, SessionStore


BOOKS = {
    "store": {
        "book": [
            {"title": "Sayings", "price": 8.95},
            {"title": "Sword", "price": 12.99},
            {"title": "Moby Dick", "price": 8.99},
        ],
        "bicycle": {"color": "red", "price": 19.95},
    }
}


@pytest.fixture()
def store(tmp_path):
    store = SessionStore(str(tmp_path / "sessions"))
    app.state.session_store = store
    return store


@pytest.fixture()
def client(store):
    return TestClient(app)


def create_session(client, document=None, version=None):
    resp = client.post(
        "/sessions", json={"document": document, "version": version or {}}
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


def submit(client, session_id, expected_revision, operations):
    return client.post(
        f"/sessions/{session_id}/batches",
        json={"expected_revision": expected_revision, "operations": operations},
    )


def replace(client, session_id, expected_revision, actor, path, value, key="path"):
    return client.post(
        f"/sessions/{session_id}/replacements",
        json={
            "expected_revision": expected_revision,
            "actor": actor,
            key: path,
            "value": value,
        },
    )


# ---------------------------------------------------------------------------
# basic selection and operation generation
# ---------------------------------------------------------------------------


def test_replacement_generates_preordered_contiguous_sets(client):
    sid = create_session(client, BOOKS)
    resp = replace(client, sid, 0, "alice", "$..price", 0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 1
    assert body["paths"] == [
        "/store/book/0/price",
        "/store/book/1/price",
        "/store/book/2/price",
        "/store/bicycle/price",
    ]
    assert body["operations"] == [
        {"actor": "alice", "seq": 1},
        {"actor": "alice", "seq": 2},
        {"actor": "alice", "seq": 3},
        {"actor": "alice", "seq": 4},
    ]
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["version"] == {"alice": 4}
    assert state["document"]["store"]["bicycle"]["price"] == 0
    assert [b["price"] for b in state["document"]["store"]["book"]] == [0, 0, 0]


def test_query_alias_is_accepted(client):
    sid = create_session(client, {"a": {"x": 1}, "b": {"x": 2}})
    resp = replace(client, sid, 0, "alice", "$..x", 9, key="query")
    assert resp.status_code == 200
    assert resp.json()["paths"] == ["/a/x", "/b/x"]


def test_filter_expression_uses_query_grammar(client):
    sid = create_session(client, BOOKS)
    resp = replace(client, sid, 0, "alice", "$.store.book[?(@.price < 10)].title", "SALE")
    assert resp.status_code == 200
    body = resp.json()
    assert body["paths"] == ["/store/book/0/title", "/store/book/2/title"]
    titles = [b["title"] for b in client.get(f"/sessions/{sid}").json()
              ["document"]["store"]["book"]]
    assert titles == ["SALE", "Sword", "SALE"]


def test_wildcard_replaces_object_members_and_array_elements(client):
    sid = create_session(client, {"a": 1, "b": 2})
    resp = replace(client, sid, 0, "zoe", "$[*]", 7)
    assert resp.status_code == 200
    assert resp.json()["paths"] == ["/a", "/b"]
    assert client.get(f"/sessions/{sid}").json()["document"] == {"a": 7, "b": 7}

    sid2 = create_session(client, {"items": [1, 2, 3]})
    resp = replace(client, sid2, 0, "zoe", "$.items[*]", 0)
    assert resp.status_code == 200
    assert resp.json()["paths"] == ["/items/0", "/items/1", "/items/2"]
    assert client.get(f"/sessions/{sid2}").json()["document"] == {"items": [0, 0, 0]}


def test_root_match_replaces_whole_document(client):
    sid = create_session(client, {"a": 1})
    resp = replace(client, sid, 0, "alice", "$", {"reset": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["paths"] == [""]
    assert body["operations"] == [{"actor": "alice", "seq": 1}]
    assert client.get(f"/sessions/{sid}").json()["document"] == {"reset": True}


def test_complex_value_is_written_to_every_location(client):
    sid = create_session(client, {"a": 1, "b": 2})
    resp = replace(client, sid, 0, "alice", "$[*]", {"nested": [1, {"x": 2}]})
    assert resp.status_code == 200
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"] == {"a": {"nested": [1, {"x": 2}]},
                                 "b": {"nested": [1, {"x": 2}]}}


# ---------------------------------------------------------------------------
# causal versions cover prior log and earlier generated operations
# ---------------------------------------------------------------------------


def test_generated_versions_cover_stored_log_and_run_on_after_existing_seqs(client):
    # prior history: alice:1 plus bob:1 (which witnessed alice:1)
    sid = create_session(client, BOOKS)
    submit(
        client, sid, 0,
        [
            {"actor": "alice", "seq": 1, "version": {"alice": 1},
             "path": "/store/bicycle/color", "action": "set", "value": "blue"},
            {"actor": "bob", "seq": 1, "version": {"alice": 1, "bob": 1},
             "path": "/store/bicycle/color", "action": "set", "value": "green"},
        ],
    )
    resp = replace(client, sid, 1, "alice", "$..price", 0)
    assert resp.status_code == 200
    body = resp.json()
    # alice continues right after her stored seq 1
    assert [o["seq"] for o in body["operations"]] == [2, 3, 4, 5]
    # a brand-new actor starts at seq 1 and causally covers alice:2..5 + bob:1
    resp2 = replace(client, sid, 2, "carol", "$..title", "x")
    assert resp2.status_code == 200
    state = client.get(f"/sessions/{sid}").json()
    store = app.state.session_store
    record = store.get(sid)
    carol = next(o for o in record.log if o.actor == "carol")
    assert carol.version == {"alice": 5, "bob": 1, "carol": 1}
    # generated ops are ordinary set ops committed under the same rules:
    # alice's prices causally supersede everyone, no conflict reported
    assert state["conflicts"] == []
    assert state["version"] == {"alice": 5, "bob": 1, "carol": 3}


# ---------------------------------------------------------------------------
# zero matches: no revision, no history
# ---------------------------------------------------------------------------


def test_zero_matches_do_not_increment_revision(client):
    sid = create_session(client, BOOKS)
    replace(client, sid, 0, "alice", "$..price", 0)
    resp = replace(client, sid, 1, "alice", "$..no_such_member", 1)
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "session_id": sid,
        "revision": 1,
        "paths": [],
        "operations": [],
    }
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["version"] == {"alice": 4}
    # the zero-match attempt added no history boundary
    changes = client.post(
        f"/sessions/{sid}/changes",
        json={"after_revision": 0,
              "subscriptions": [{"subscriber_id": "all", "include": ["/**"]}]},
    ).json()
    assert [c["revision"] for c in changes["changes"]] == [1]


# ---------------------------------------------------------------------------
# 20-match limit and parent/child nesting
# ---------------------------------------------------------------------------


def _array_doc(n):
    return {"items": list(range(n))}


def test_twenty_matches_are_accepted(client):
    sid = create_session(client, _array_doc(20))
    resp = replace(client, sid, 0, "alice", "$.items[*]", "x")
    assert resp.status_code == 200
    assert len(resp.json()["paths"]) == 20
    assert len(resp.json()["operations"]) == 20
    assert resp.json()["revision"] == 1


def test_more_than_twenty_matches_rejected_without_writing(client):
    sid = create_session(client, _array_doc(21))
    resp = replace(client, sid, 0, "alice", "$.items[*]", "x")
    assert resp.status_code == 422
    assert "at most 20" in resp.json()["detail"]
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 0
    assert state["document"] == _array_doc(21)


def test_parent_and_child_matches_rejected(client):
    sid = create_session(client, {"a": {"b": {"c": 1}}})
    # recursive wildcard selects the subtree /a, /a/b, /a/b/c together
    resp = replace(client, sid, 0, "alice", "$..*", 0)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "nested" in detail and "contains" in detail
    assert client.get(f"/sessions/{sid}").json()["revision"] == 0


def test_sibling_paths_with_prefix_like_names_are_not_nested(client):
    # "/a" and "/ab" share a string prefix but are not parent/child
    sid = create_session(client, {"a": 1, "ab": 2})
    resp = replace(client, sid, 0, "alice", "$[*]", 0)
    assert resp.status_code == 200
    assert resp.json()["paths"] == ["/a", "/ab"]


# ---------------------------------------------------------------------------
# invalid input -> 422, nothing written
# ---------------------------------------------------------------------------


def test_invalid_jsonpath_returns_422_and_leaves_session_untouched(client):
    sid = create_session(client, {"a": 1})
    resp = replace(client, sid, 0, "alice", "$.a[0:2]", 0)
    assert resp.status_code == 422
    assert "invalid JSONPath" in resp.json()["detail"]
    assert client.get(f"/sessions/{sid}").json()["revision"] == 0


def test_validation_error_writes_no_operations_or_history(client):
    sid = create_session(client, _array_doc(21))
    replace(client, sid, 0, "alice", "$.items[*]", "x")  # 422
    changes = client.post(
        f"/sessions/{sid}/changes",
        json={"after_revision": 0,
              "subscriptions": [{"subscriber_id": "all", "include": ["/**"]}]},
    ).json()
    assert changes["changes"] == []
    assert changes["current_revision"] == 0


# ---------------------------------------------------------------------------
# optimistic concurrency
# ---------------------------------------------------------------------------


def test_stale_expected_revision_returns_409(client):
    sid = create_session(client, BOOKS)
    replace(client, sid, 0, "alice", "$..price", 0)
    resp = replace(client, sid, 0, "bob", "$..title", "x")
    assert resp.status_code == 409
    # the loser's paths were never written
    state = client.get(f"/sessions/{sid}").json()
    assert state["version"] == {"alice": 4}
    assert state["document"]["store"]["book"][0]["title"] == "Sayings"


def test_future_expected_revision_returns_409(client):
    sid = create_session(client, BOOKS)
    resp = replace(client, sid, 3, "alice", "$..price", 0)
    assert resp.status_code == 409
    assert client.get(f"/sessions/{sid}").json()["revision"] == 0


def test_unknown_session_returns_404(client):
    missing = "0" * 32
    resp = replace(client, missing, 0, "alice", "$", 1)
    assert resp.status_code == 404


def test_concurrent_same_revision_only_one_replacement_commits(tmp_path):
    store = SessionStore(str(tmp_path / "sessions"))
    rec = store.create({"items": list(range(8))}, {})

    def attempt(i):
        try:
            return store.replace_matches(
                rec.session_id, 0, f"actor{i}", "$.items[*]", i
            )
        except RevisionConflictError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(12)))

    committed = [o for o in outcomes if o is not None]
    assert len(committed) == 1
    assert committed[0].revision == 1
    assert len(committed[0].operations) == 8
    record = store.get(rec.session_id)
    assert record.revision == 1
    assert {o.actor for o in record.log} == {committed[0].operations[0][0]}


# ---------------------------------------------------------------------------
# history: the commit is visible to /changes, unchanged docs keep the revision
# ---------------------------------------------------------------------------


def test_changes_include_the_replacement_commit(client):
    sid = create_session(client, {"items": [1, 2, 3]})
    replace(client, sid, 0, "alice", "$.items[*]", 0)
    changes = client.post(
        f"/sessions/{sid}/changes",
        json={"after_revision": 0,
              "subscriptions": [{"subscriber_id": "items", "include": ["/items"]}]},
    ).json()
    assert changes["current_revision"] == 1
    assert changes["next_cursor"] == 1
    rev1 = changes["changes"][0]
    assert rev1["revision"] == 1
    events = rev1["routes"][0]["events"]
    assert events[0] == {"type": "replace", "path": "/items",
                         "old": [1, 2, 3], "new": [0, 0, 0]}


def test_replacement_with_no_document_change_still_appends_empty_revision(client):
    sid = create_session(client, {"a": 1, "b": 1})
    resp = replace(client, sid, 0, "alice", "$[*]", 1)  # writes the existing value
    assert resp.status_code == 200
    assert resp.json()["revision"] == 1
    changes = client.post(
        f"/sessions/{sid}/changes",
        json={"after_revision": 0,
              "subscriptions": [{"subscriber_id": "all", "include": ["/**"]}]},
    ).json()
    assert [c["revision"] for c in changes["changes"]] == [1]
    rev1 = changes["changes"][0]
    assert rev1["routes"][0]["events"] == []
    assert rev1["unrouted"] == []
    # the operations really are in the log even though the document did not move
    state = client.get(f"/sessions/{sid}").json()
    assert state["version"] == {"alice": 2}


# ---------------------------------------------------------------------------
# durability and cross-process safety
# ---------------------------------------------------------------------------


def test_replacement_survives_store_restart(tmp_path):
    store_dir = str(tmp_path / "sessions")
    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = create_session(client, BOOKS)
    resp = replace(client, sid, 0, "alice", "$..price", 0)
    assert resp.json()["revision"] == 1

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["version"] == {"alice": 4}
    # a stale replacement conflicts after the restart as well
    assert replace(client, sid, 0, "bob", "$..title", "x").status_code == 409
    # and history remains pageable from the persisted boundary
    resp = client.post(
        f"/sessions/{sid}/changes",
        json={"after_revision": 1,
              "subscriptions": [{"subscriber_id": "all", "include": ["/**"]}]},
    )
    assert resp.status_code == 200
    assert resp.json()["changes"] == []


def _racing_replacer(store_dir, session_id, actor, barrier, queue):
    """Worker: read revision 0, wait for the peer, then replace by JSONPath."""
    store = SessionStore(store_dir)
    assert store.get(session_id).revision == 0
    barrier.wait(timeout=15)
    try:
        store.replace_matches(session_id, 0, actor, "$.items[*]", actor)
        status = "committed"
    except RevisionConflictError:
        status = "conflict"
    state = store.get_state(session_id)
    queue.put((status, actor, state["revision"], state["document"]))


def test_shared_store_dir_two_processes_only_one_replacement_commits(tmp_path):
    store_dir = str(tmp_path / "sessions")
    rec = SessionStore(store_dir).create({"items": [1, 2, 3]}, {})

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_racing_replacer,
            args=(store_dir, rec.session_id, f"actor{i}", barrier, queue),
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0

    results = [queue.get(timeout=10) for _ in procs]
    assert sorted(r[0] for r in results) == ["committed", "conflict"]
    winner = next(r[1] for r in results if r[0] == "committed")
    for _, _, revision, document in results:
        assert revision == 1
        assert document == {"items": [winner, winner, winner]}
