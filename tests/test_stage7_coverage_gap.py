import json
from pathlib import Path

import pytest

from benchmarks import stage7_coverage_gap as coverage_gap


@pytest.fixture()
def sample_matrix(tmp_path: Path) -> Path:
    matrix = {
        "stage": "S7",
        "gate": "G7",
        "l1": [
            {
                "op": "relu",
                "shape_tags_required": ["shapeA", "shapeB"],
                "shape_count_required": 2,
            },
            {
                "op": "multi_latent_attention",
                "shape_tags_required": ["shapeX"],
                "shape_count_required": 1,
            },
        ],
        "l2": [
            {
                "op": "matmul_relu",
                "shape_tags_required": ["shapeC", "shapeD"],
                "shape_count_required": 2,
            }
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    return path


@pytest.fixture()
def results_root(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    (root / "l1").mkdir(parents=True)
    (root / "l2").mkdir(parents=True)

    (root / "l1" / "PERF_ALL.csv").write_text(
        """operator,shape_tag,correctness_status,allclose,rtol,atol,perf_target,perf_actual,perf_pass,perf_gap
relu,shapeA,pass,true,1e-5,1e-6,1.0,1.1,true,0.0
relu,shapeB,pass,true,1e-5,1e-6,1.0,0.9,false,-0.1
""".strip()
    )

    (root / "l2" / "PERF_ALL.csv").write_text(
        """operator,shape_tag,perf_target,perf_actual,perf_pass,perf_gap
matmul_relu,shapeC,1.0,1.05,true,0.0
""".strip()
    )

    return root


def test_compute_gap_counts_shapes_and_fields(sample_matrix: Path, results_root: Path):
    report = coverage_gap.compute_gap(sample_matrix, results_root)

    assert report["stage"] == "S7"
    assert report["gate"] == "G7"
    assert report["l1"]["ops_total"] == 2
    assert report["l1"]["ops_with_any_evidence"] == 1
    assert report["l1"]["shapes_observed_total"] == 2
    assert report["l1"]["shapes_required_total"] == 3

    l1_entries = {entry["op"]: entry for entry in report["l1"]["per_op"]}
    relu = l1_entries["relu"]
    assert relu["observed_shape_count"] == 2
    assert relu["missing_shape_tags"] == []
    assert relu["correctness_fields_present"] is True
    assert relu["perf_target_fields_present"] is True

    mla = l1_entries["multi_latent_attention"]
    assert mla["observed_shape_count"] == 0
    assert mla["missing_shape_tags"] == ["shapeX"]
    assert mla["correctness_fields_present"] is False
    assert mla["perf_target_fields_present"] is False
    assert set(mla["missing_perf_target_fields"]) == {
        "perf_target",
        "perf_actual",
        "perf_pass",
        "perf_gap",
    }

    l2_entries = {entry["op"]: entry for entry in report["l2"]["per_op"]}
    mm_relu = l2_entries["matmul_relu"]
    assert mm_relu["observed_shape_count"] == 1
    assert mm_relu["missing_shape_tags"] == ["shapeD"]
    assert mm_relu["correctness_fields_present"] is False
    assert mm_relu["perf_target_fields_present"] is True

    text_summary = coverage_gap.format_text_summary(report)
    assert "[L1] ops 1/2" in text_summary
    assert "multi_latent_attention" in text_summary
    assert "matmul_relu" in text_summary


def test_compute_gap_handles_raw_op_column(sample_matrix: Path, tmp_path: Path):
    root = tmp_path / "results"
    (root / "l1").mkdir(parents=True)
    (root / "l2").mkdir(parents=True)

    (root / "l1" / "PERF_ALL.csv").write_text(
        """op,shape_tag,correctness_status,allclose,rtol,atol,perf_target,perf_actual,perf_pass,perf_gap
relu,shapeA,pass,true,1e-5,1e-6,1.0,1.1,true,0.1
""".strip()
    )
    (root / "l2" / "PERF_ALL.csv").write_text(
        """op,shape_tag,perf_target,perf_actual,perf_pass,perf_gap
matmul_relu,shapeC,1.0,1.05,true,0.05
""".strip()
    )

    report = coverage_gap.compute_gap(sample_matrix, root)

    l1_entries = {entry["op"]: entry for entry in report["l1"]["per_op"]}
    assert l1_entries["relu"]["observed_shape_tags"] == ["shapeA"]
    assert l1_entries["relu"]["correctness_fields_present"] is True
    assert l1_entries["relu"]["perf_target_fields_present"] is True

    l2_entries = {entry["op"]: entry for entry in report["l2"]["per_op"]}
    assert l2_entries["matmul_relu"]["observed_shape_tags"] == ["shapeC"]
    assert l2_entries["matmul_relu"]["perf_target_fields_present"] is True


def test_compute_gap_handles_missing_perf_files(sample_matrix: Path, tmp_path: Path):
    report = coverage_gap.compute_gap(sample_matrix, tmp_path / "missing")
    assert report["l1"]["ops_with_any_evidence"] == 0
    assert report["l1"]["shapes_observed_total"] == 0
    assert report["l2"]["ops_with_any_evidence"] == 0
    assert report["l2"]["shapes_observed_total"] == 0
