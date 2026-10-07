"""The numpy-only reader for saved LightGBM text models, used when
LightGBM's native library cannot be loaded."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.classification.lightgbm_classifier import LightGBMClassifier
from src.classification.lightgbm_text_model import TextModelBooster, UnsupportedModelError
from src.core.models import FeatureVector
from src.features.feature_vector import FEATURE_NAMES

LIVE_MODEL = Path(__file__).resolve().parents[1] / "models" / "lightgbm" / "person_classifier.txt"

# Two trees over 3 features, small enough to score by hand.
#   Tree 0: f0 <= 0.5 ? (f1 <= 2.5 ? leaf0 : leaf1) : leaf2      (missing: none)
#   Tree 1: f2 is zero-as-missing with default RIGHT; else f2 <= 1.0 ? leaf0 : leaf1
HAND_MODEL = """tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=2
objective=binary sigmoid:1
feature_names=Column_0 Column_1 Column_2
feature_infos=[0:1] [0:5] [0:3]
tree_sizes=300 300

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=10 5
threshold=0.5 2.5
decision_type=2 2
left_child=1 -1
right_child=-3 -2
leaf_value=0.1 0.2 -0.3
leaf_weight=1 1 1
leaf_count=1 1 1
internal_value=0 0
internal_weight=3 2
internal_count=3 2
is_linear=0
shrinkage=1


Tree=1
num_leaves=2
num_cat=0
split_feature=2
split_gain=4
threshold=1.0
decision_type=4
left_child=-1
right_child=-2
leaf_value=0.5 -0.5
leaf_weight=1 1
leaf_count=1 1
internal_value=0
internal_weight=2
internal_count=2
is_linear=0
shrinkage=1


end of trees

feature_importances:
Column_0=1

parameters:
[boosting: gbdt]
end of parameters
"""


def _sigmoid(raw: float) -> float:
    return 1.0 / (1.0 + math.exp(-raw))


@pytest.mark.parametrize("row, raw", [
    ([0.0, 1.0, 0.5], 0.1 + 0.5),      # tree 0 left-left; tree 1 below threshold
    ([0.0, 3.0, 2.0], 0.2 - 0.5),      # tree 0 left-right; tree 1 above threshold
    ([1.0, 0.0, 1.0], -0.3 + 0.5),     # tree 0 right; threshold itself goes left
    ([0.5, 2.5, 0.0], 0.1 - 0.5),      # both thresholds exact; f2 zero is "missing" -> default right
    ([float("nan"), 9.0, float("nan")], 0.2 - 0.5),  # NaN reads as 0.0, then zero-as-missing -> right
])
def test_hand_scored_rows(row, raw):
    booster = TextModelBooster(HAND_MODEL)
    assert booster.num_feature() == 3 and booster.num_trees() == 2
    assert booster.predict([row])[0] == pytest.approx(_sigmoid(raw), abs=1e-12)


def test_small_and_large_batches_give_identical_scores():
    """Few rows take a plain Python walk, many rows take the numpy walk -
    the two must agree to the last bit."""
    booster = TextModelBooster.from_file(LIVE_MODEL)
    rng = np.random.default_rng(7)
    rows = rng.integers(0, 4, size=(400, booster.num_feature())).astype(float)
    rows[:, 6:12] = rng.random((400, 6))
    rows[::17, 3] = np.nan
    together = booster.predict(rows)
    one_by_one = np.array([booster.predict(rows[i:i + 1])[0] for i in range(len(rows))])
    assert np.array_equal(together, one_by_one)
    assert ((together > 0.0) & (together < 1.0)).all()


def test_wrong_column_count_is_refused():
    booster = TextModelBooster(HAND_MODEL)
    with pytest.raises(ValueError, match="number of features"):
        booster.predict([[0.0, 1.0]])


def test_truncated_and_unsupported_models_are_refused():
    with pytest.raises(ValueError, match="truncated"):
        TextModelBooster(HAND_MODEL.split("end of trees")[0])
    with pytest.raises(ValueError, match="holds 1 trees"):
        TextModelBooster(HAND_MODEL.replace(HAND_MODEL[HAND_MODEL.index("Tree=1"):HAND_MODEL.index("end of trees")], ""))
    with pytest.raises(UnsupportedModelError, match="categorical"):
        TextModelBooster(HAND_MODEL.replace("decision_type=4", "decision_type=1"))
    with pytest.raises(UnsupportedModelError, match="Multi-class"):
        TextModelBooster(HAND_MODEL.replace("num_class=1", "num_class=3"))


def test_save_model_writes_the_same_model_back(tmp_path):
    booster = TextModelBooster.from_file(LIVE_MODEL)
    copy = tmp_path / "copy.txt"
    booster.save_model(copy)
    rows = np.random.default_rng(1).integers(0, 3, size=(20, booster.num_feature())).astype(float)
    assert np.array_equal(TextModelBooster.from_file(copy).predict(rows), booster.predict(rows))


def test_classifier_falls_back_when_lightgbm_cannot_be_loaded(monkeypatch):
    """The 2026-10-06 failure: importing lightgbm raises OSError (the OS
    blocks its DLL). The classifier must still load the model and score."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "lightgbm":
            raise OSError("[WinError 4551] An Application Control policy has blocked this file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    classifier = LightGBMClassifier()
    classifier.load(LIVE_MODEL)
    assert classifier.backend == "text_model_reader"

    values = [1] * 19
    vector = FeatureVector(*values, common_word_ratio=0.5, first_token_is_ambiguous=0, place_or_org_token_count=0)
    assert len(vector.as_list()) == len(FEATURE_NAMES)
    single = classifier.predict_proba(vector)
    batch = classifier.predict_proba_batch([vector] * 12)
    assert single.ok and all(b.ok for b in batch)
    assert {b.probability for b in batch} == {single.probability}
    assert 0.0 < single.probability < 1.0


def test_reader_matches_lightgbm_itself(tmp_path):
    """Wherever the real library loads, the reader must reproduce its
    scores on the live model and on a freshly trained one."""
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as exc:
        pytest.skip(f"LightGBM's native library cannot be loaded here: {exc}")

    rng = np.random.default_rng(3)
    live = lgb.Booster(model_file=str(LIVE_MODEL))
    rows = rng.integers(0, 4, size=(500, live.num_feature())).astype(float)
    rows[:, 6:12] = rng.random((500, 6))
    rows[::13, 2] = np.nan
    assert np.allclose(TextModelBooster.from_file(LIVE_MODEL).predict(rows), live.predict(rows), rtol=0, atol=1e-12)

    X = rng.random((600, 8))
    X[rng.random((600, 8)) < 0.1] = 0.0
    X[rng.random((600, 8)) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 3]) > 0.9).astype(int)
    for params in ({"use_missing": True}, {"zero_as_missing": True}, {"use_missing": False}):
        trained = lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5, **params},
                            lgb.Dataset(X, y), num_boost_round=20)
        saved = tmp_path / "trained.txt"
        trained.save_model(str(saved))
        assert np.allclose(TextModelBooster.from_file(saved).predict(X), trained.predict(X), rtol=0, atol=1e-12)
