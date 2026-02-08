"""Tests for ModelAssignments persistence."""
from __future__ import annotations

import json
import pytest

from gldpred.registry.assignments import ModelAssignments


class TestModelAssignments:

    def test_assign_and_get(self, tmp_path):
        assignments = ModelAssignments(state_dir=tmp_path)
        assignments.assign("GLD", "model_abc")
        assert assignments.get("GLD") == "model_abc"

    def test_unassign(self, tmp_path):
        assignments = ModelAssignments(state_dir=tmp_path)
        assignments.assign("GLD", "model_abc")
        assignments.unassign("GLD")
        assert assignments.get("GLD") is None

    def test_get_all(self, tmp_path):
        assignments = ModelAssignments(state_dir=tmp_path)
        assignments.assign("GLD", "model_1")
        assignments.assign("SLV", "model_2")
        result = assignments.get_all()
        assert result == {"GLD": "model_1", "SLV": "model_2"}

    def test_reset(self, tmp_path):
        assignments = ModelAssignments(state_dir=tmp_path)
        assignments.assign("GLD", "model_1")
        assignments.assign("SLV", "model_2")
        assignments.reset()
        assert assignments.get_all() == {}

    def test_has(self, tmp_path):
        assignments = ModelAssignments(state_dir=tmp_path)
        assert not assignments.has("GLD")
        assignments.assign("GLD", "model_1")
        assert assignments.has("GLD")

    def test_persistence_across_instances(self, tmp_path):
        a1 = ModelAssignments(state_dir=tmp_path)
        a1.assign("BTC-USD", "btc_model")
        a2 = ModelAssignments(state_dir=tmp_path)
        assert a2.get("BTC-USD") == "btc_model"

    def test_json_file_created(self, tmp_path):
        assignments = ModelAssignments(state_dir=tmp_path)
        assignments.assign("PALL", "pall_model")
        json_file = tmp_path / "asset_model_map.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert data["PALL"] == "pall_model"

    def test_overwrite_assignment(self, tmp_path):
        assignments = ModelAssignments(state_dir=tmp_path)
        assignments.assign("GLD", "model_v1")
        assignments.assign("GLD", "model_v2")
        assert assignments.get("GLD") == "model_v2"

    def test_unassign_nonexistent(self, tmp_path):
        """Unassigning a non-existent ticker does not raise."""
        assignments = ModelAssignments(state_dir=tmp_path)
        assignments.unassign("FAKE")  # Should not raise
        assert assignments.get("FAKE") is None
