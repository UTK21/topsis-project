"""Tests for the TOPSIS command-line implementation in ``topsis/__main__.py``.

``main()`` reads its inputs from ``sys.argv``, so each test patches
``sys.argv`` and checks the CSV file it writes (or how it fails).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topsis.__main__ import main

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "data.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def reference_topsis(matrix, weights, impacts):
    """Independent, vectorised TOPSIS used as an oracle for the CLI."""
    m = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    norm = m / np.sqrt((m ** 2).sum(axis=0))
    weighted = norm * w
    benefit = np.array([i == "+" for i in impacts])
    ideal_best = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    ideal_worst = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst)


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["topsis", *[str(a) for a in args]])
    main()


@pytest.fixture
def run_topsis(tmp_path, monkeypatch):
    """Run the CLI on a DataFrame (or CSV path) and return the result DataFrame."""

    def _run(data, weights, impacts):
        if isinstance(data, pd.DataFrame):
            input_path = tmp_path / "input.csv"
            data.to_csv(input_path, index=False)
        else:
            input_path = data
        output_path = tmp_path / "result.csv"
        run_cli(monkeypatch, input_path, weights, impacts, output_path)
        return pd.read_csv(output_path, index_col=0)

    return _run


@pytest.fixture
def sample_df():
    return pd.read_csv(SAMPLE_CSV)


# --------------------------------------------------------------------------- #
# Correctness of the computed scores and ranks
# --------------------------------------------------------------------------- #
SAMPLE_WEIGHTS = "1,1,1,1,1"
SAMPLE_IMPACTS = "+,+,-,+,+"


def test_sample_scores_match_reference(run_topsis, sample_df):
    result = run_topsis(SAMPLE_CSV, SAMPLE_WEIGHTS, SAMPLE_IMPACTS)
    expected = reference_topsis(
        sample_df.iloc[:, 1:].values, [1] * 5, SAMPLE_IMPACTS.split(",")
    )
    np.testing.assert_allclose(result["PerformanceScore"].values, expected)


def test_sample_known_ranking(run_topsis):
    result = run_topsis(SAMPLE_CSV, SAMPLE_WEIGHTS, SAMPLE_IMPACTS)
    ranking = dict(zip(result["Fund Name"], result["Rank"]))
    assert ranking == {
        "M1": 3, "M2": 7, "M3": 8, "M4": 4,
        "M5": 5, "M6": 2, "M7": 1, "M8": 6,
    }


def test_output_preserves_input_and_adds_columns(run_topsis, sample_df):
    result = run_topsis(SAMPLE_CSV, SAMPLE_WEIGHTS, SAMPLE_IMPACTS)
    assert list(result.columns) == list(sample_df.columns) + [
        "Index", "PerformanceScore", "Rank",
    ]
    pd.testing.assert_frame_equal(
        result[sample_df.columns].reset_index(drop=True), sample_df
    )
    # Rows stay in input order (not sorted by rank).
    assert list(result.index) == list(range(len(sample_df)))
    assert list(result["Index"]) == list(range(1, len(sample_df) + 1))


def test_scores_between_zero_and_one(run_topsis):
    result = run_topsis(SAMPLE_CSV, SAMPLE_WEIGHTS, SAMPLE_IMPACTS)
    assert result["PerformanceScore"].between(0, 1).all()


def test_ranks_are_a_permutation_ordered_by_score(run_topsis):
    result = run_topsis(SAMPLE_CSV, SAMPLE_WEIGHTS, SAMPLE_IMPACTS)
    n = len(result)
    assert sorted(result["Rank"]) == list(range(1, n + 1))
    by_rank = result.sort_values("Rank")["PerformanceScore"].values
    assert np.all(np.diff(by_rank) <= 0), "higher score must get a better rank"


@pytest.mark.parametrize(
    "weights, impacts",
    [
        ("1,1,1,1,1", "+,+,+,+,+"),
        ("1,1,1,1,1", "-,-,-,-,-"),
        ("2,1,3,1,1", "+,+,-,+,+"),
        ("5,1,1,1,1", "-,+,-,+,-"),
    ],
)
def test_matches_reference_for_various_weights_and_impacts(
    run_topsis, sample_df, weights, impacts
):
    result = run_topsis(SAMPLE_CSV, weights, impacts)
    expected = reference_topsis(
        sample_df.iloc[:, 1:].values,
        [int(w) for w in weights.split(",")],
        impacts.split(","),
    )
    np.testing.assert_allclose(result["PerformanceScore"].values, expected)


def test_dominant_alternative_ranks_first(run_topsis):
    df = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Gain": [10.0, 5.0, 1.0],   # higher is better
            "Cost": [1.0, 5.0, 10.0],   # lower is better
        }
    )
    result = run_topsis(df, "1,1", "+,-")
    assert list(result["Rank"]) == [1, 2, 3]
    assert result["PerformanceScore"].iloc[0] == pytest.approx(1.0)
    assert result["PerformanceScore"].iloc[-1] == pytest.approx(0.0)


def test_flipping_all_impacts_reverses_ranking(run_topsis):
    df = pd.DataFrame(
        {
            "Model": ["A", "B", "C", "D"],
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [2.0, 3.0, 5.0, 7.0],
        }
    )
    plus = run_topsis(df, "1,1", "+,+")
    minus = run_topsis(df, "1,1", "-,-")
    assert list(plus["Rank"]) == [4, 3, 2, 1]
    assert list(minus["Rank"]) == [1, 2, 3, 4]
    np.testing.assert_allclose(
        plus["PerformanceScore"].values, 1 - minus["PerformanceScore"].values
    )


def test_weights_change_the_winner(run_topsis):
    # A is best on X, B is best on Y; the heavier criterion decides.
    df = pd.DataFrame(
        {"Model": ["A", "B"], "X": [9.0, 1.0], "Y": [1.0, 9.0]}
    )
    favour_x = run_topsis(df, "5,1", "+,+")
    favour_y = run_topsis(df, "1,5", "+,+")
    assert list(favour_x["Rank"]) == [1, 2]
    assert list(favour_y["Rank"]) == [2, 1]


def test_scaling_a_criterion_does_not_change_scores(run_topsis):
    df = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "X": [1.0, 2.0, 3.0],
            "Y": [3.0, 1.0, 2.0],
        }
    )
    scaled = df.copy()
    scaled["X"] = scaled["X"] * 1000
    np.testing.assert_allclose(
        run_topsis(df, "1,1", "+,-")["PerformanceScore"].values,
        run_topsis(scaled, "1,1", "+,-")["PerformanceScore"].values,
    )


@pytest.mark.xfail(
    reason="normalised floats are written back into int64 columns in place; "
    "pandas >= 3 rejects this with TypeError (older pandas only warns)",
    raises=TypeError,
    strict=False,
)
def test_integer_valued_criteria(run_topsis):
    df = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "X": [1, 2, 3],
            "Y": [30, 10, 20],
        }
    )
    result = run_topsis(df, "1,1", "+,-")
    expected = reference_topsis(df.iloc[:, 1:].values, [1, 1], ["+", "-"])
    np.testing.assert_allclose(result["PerformanceScore"].values, expected)


def test_prints_input_and_result(run_topsis, capsys):
    run_topsis(SAMPLE_CSV, SAMPLE_WEIGHTS, SAMPLE_IMPACTS)
    out = capsys.readouterr().out
    assert "Fund Name" in out
    assert "PerformanceScore" in out
    assert "Rank" in out


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "args",
    [
        [],
        ["data.csv"],
        ["data.csv", "1,1"],
        ["data.csv", "1,1", "+,+"],
        ["data.csv", "1,1", "+,+", "out.csv", "extra"],
    ],
)
def test_wrong_number_of_arguments(monkeypatch, capsys, args):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, *args)
    assert "Number of arguements is invalid" in capsys.readouterr().out


def test_missing_input_file(monkeypatch, capsys, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_cli(monkeypatch, tmp_path / "nope.csv", "1,1", "+,+", tmp_path / "o.csv")
    assert "does not exist" in capsys.readouterr().out


def test_too_few_columns(monkeypatch, capsys, tmp_path):
    path = tmp_path / "two_cols.csv"
    pd.DataFrame({"Model": ["A", "B"], "X": [1, 2]}).to_csv(path, index=False)
    out_path = tmp_path / "o.csv"
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, path, "1", "+", out_path)
    assert "Number of columns" in capsys.readouterr().out
    assert not out_path.exists()


@pytest.mark.parametrize(
    "weights, impacts, message",
    [
        # weights and impacts disagree in length
        ("1,1,1,1,1", "+,+,+,+", "Inapprpriate Dimensions1"),
        ("1,1,1,1", "+,+,+,+,+", "Inapprpriate Dimensions1"),
        # same length as each other, but not the number of criteria (5)
        ("1,1,1,1", "+,+,+,+", "Inapprpriate Dimensions2"),
        ("1,1,1,1,1,1", "+,+,+,+,+,+", "Inapprpriate Dimensions2"),
    ],
)
def test_dimension_mismatch(monkeypatch, capsys, tmp_path, weights, impacts, message):
    out_path = tmp_path / "o.csv"
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, SAMPLE_CSV, weights, impacts, out_path)
    assert message in capsys.readouterr().out
    assert not out_path.exists()


@pytest.mark.parametrize("impacts", ["+,+,*,+,+", "+,+,-,+,x", "+,+,-,+,++", "+,+,-,+,"])
def test_invalid_impact_symbol(monkeypatch, capsys, tmp_path, impacts):
    out_path = tmp_path / "o.csv"
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, SAMPLE_CSV, "1,1,1,1,1", impacts, out_path)
    assert "Imapct can be either + or - only" in capsys.readouterr().out
    assert not out_path.exists()


# --------------------------------------------------------------------------- #
# Known defects (documented as expected failures)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason="weights are parsed with int(), so fractional weights raise ValueError",
    raises=ValueError,
    strict=True,
)
def test_fractional_weights(run_topsis, sample_df):
    weights = "0.5,1,1,1,1.5"
    result = run_topsis(SAMPLE_CSV, weights, SAMPLE_IMPACTS)
    expected = reference_topsis(
        sample_df.iloc[:, 1:].values,
        [float(w) for w in weights.split(",")],
        SAMPLE_IMPACTS.split(","),
    )
    np.testing.assert_allclose(result["PerformanceScore"].values, expected)


@pytest.mark.xfail(
    reason="setup.py points the console script at topsis.topsis:main, "
    "but there is no topsis/topsis.py module",
    raises=ModuleNotFoundError,
    strict=True,
)
def test_console_script_entry_point_is_importable():
    import importlib
    import re

    setup_py = (REPO_ROOT / "setup.py").read_text()
    target = re.search(r'"topsis\s*=\s*([\w.]+):(\w+)"', setup_py)
    assert target, "console_scripts entry for 'topsis' not found"
    module = importlib.import_module(target.group(1))
    assert callable(getattr(module, target.group(2)))


def test_package_version_matches_setup():
    import re

    import topsis

    setup_py = (REPO_ROOT / "setup.py").read_text()
    assert re.search(r'version="([^"]+)"', setup_py).group(1) == topsis.__version__
