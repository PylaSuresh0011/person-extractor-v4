"""
src.classification.lightgbm_text_model
=========================================
Scores rows from a saved LightGBM text model with numpy alone, for
machines where LightGBM's native library cannot be loaded. Found
2026-10-06: Windows Smart App Control started blocking lib_lightgbm.dll,
every model load failed, and the pipeline dropped to rules-only scoring
(benchmark accepted recall 0.906 -> 0.839) with nothing but a log line
to show for it.

Prediction only - training and retraining still need LightGBM itself.

Covers what this project's models use: numerical splits and a binary
(sigmoid) or plain regression objective. Categorical splits, linear
trees and multi-class models are refused at load time rather than
scored wrongly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# LightGBM's kZeroThreshold (1e-35f widened to a double) - the band a
# zero-as-missing split treats as "zero".
_ZERO_THRESHOLD = 1.0000000180025095e-35
_CATEGORICAL_MASK = 1
_DEFAULT_LEFT_MASK = 2
_MISSING_ZERO = 1
_MISSING_NAN = 2
# Below this many rows a plain Python walk beats numpy's per-call cost.
_SCALAR_ROW_LIMIT = 8


class UnsupportedModelError(ValueError):
    """The model file uses a LightGBM feature this reader does not score."""


def _ints(value: str) -> list[int]:
    return [int(v) for v in value.split()]


def _floats(value: str) -> list[float]:
    return [float(v) for v in value.split()]


class _Tree:
    __slots__ = ("split_feature", "threshold", "decision_type", "left", "right", "leaf_value",
                 "split_feature_arr", "threshold_arr", "decision_type_arr", "left_arr", "right_arr",
                 "leaf_value_arr")

    def __init__(self, index: int, fields: dict[str, str], num_features: int) -> None:
        num_leaves = int(fields["num_leaves"])
        if int(fields.get("num_cat", "0")) != 0:
            raise UnsupportedModelError(f"Tree {index} has categorical splits")
        if int(fields.get("is_linear", "0")) != 0:
            raise UnsupportedModelError(f"Tree {index} is a linear tree")

        self.leaf_value = _floats(fields["leaf_value"])
        if num_leaves > 1:
            self.split_feature = _ints(fields["split_feature"])
            self.threshold = _floats(fields["threshold"])
            self.decision_type = _ints(fields["decision_type"])
            self.left = _ints(fields["left_child"])
            self.right = _ints(fields["right_child"])
        else:
            self.split_feature, self.threshold, self.decision_type, self.left, self.right = [], [], [], [], []

        splits = num_leaves - 1
        per_split = (self.split_feature, self.threshold, self.decision_type, self.left, self.right)
        if len(self.leaf_value) != num_leaves or any(len(values) != splits for values in per_split):
            raise ValueError(f"Tree {index} is malformed: field lengths do not match num_leaves={num_leaves}")
        if any(d & _CATEGORICAL_MASK for d in self.decision_type):
            raise UnsupportedModelError(f"Tree {index} has categorical splits")
        if any(not 0 <= f < num_features for f in self.split_feature):
            raise ValueError(f"Tree {index} splits on a feature outside 0..{num_features - 1}")
        if any(not -num_leaves <= child < splits for child in self.left + self.right):
            raise ValueError(f"Tree {index} has a child index outside the tree")

        self.split_feature_arr = np.array(self.split_feature, dtype=np.int64)
        self.threshold_arr = np.array(self.threshold, dtype=np.float64)
        self.decision_type_arr = np.array(self.decision_type, dtype=np.int64)
        self.left_arr = np.array(self.left, dtype=np.int64)
        self.right_arr = np.array(self.right, dtype=np.int64)
        self.leaf_value_arr = np.array(self.leaf_value, dtype=np.float64)

    def output_for_row(self, row: list[float]) -> float:
        if not self.split_feature:
            return self.leaf_value[0]
        node = 0
        while node >= 0:
            fval = row[self.split_feature[node]]
            decision = self.decision_type[node]
            missing = (decision >> 2) & 3
            is_nan = fval != fval
            if is_nan and missing != _MISSING_NAN:
                fval, is_nan = 0.0, False
            if (missing == _MISSING_ZERO and -_ZERO_THRESHOLD <= fval <= _ZERO_THRESHOLD) or \
                    (missing == _MISSING_NAN and is_nan):
                go_left = bool(decision & _DEFAULT_LEFT_MASK)
            else:
                go_left = fval <= self.threshold[node]
            node = self.left[node] if go_left else self.right[node]
        return self.leaf_value[~node]

    def output_for_rows(self, rows: np.ndarray) -> np.ndarray:
        count = rows.shape[0]
        if not self.split_feature:
            return np.full(count, self.leaf_value[0], dtype=np.float64)
        node = np.zeros(count, dtype=np.int64)
        active = np.arange(count)
        while active.size:
            current = node[active]
            fval = rows[active, self.split_feature_arr[current]]
            decision = self.decision_type_arr[current]
            missing = (decision >> 2) & 3
            is_nan = np.isnan(fval)
            fval = np.where(is_nan & (missing != _MISSING_NAN), 0.0, fval)
            use_default = ((missing == _MISSING_ZERO) & (np.abs(fval) <= _ZERO_THRESHOLD)) | \
                          ((missing == _MISSING_NAN) & is_nan)
            with np.errstate(invalid="ignore"):
                below = fval <= self.threshold_arr[current]
            go_left = np.where(use_default, (decision & _DEFAULT_LEFT_MASK) != 0, below)
            following = np.where(go_left, self.left_arr[current], self.right_arr[current])
            node[active] = following
            active = active[following >= 0]
        return self.leaf_value_arr[~node]


class TextModelBooster:
    """The slice of lightgbm.Booster that LightGBMClassifier calls on a
    loaded model: predict(), num_feature() and save_model()."""

    def __init__(self, model_text: str) -> None:
        self._model_text = model_text
        lines = model_text.splitlines()
        if "end of trees" not in lines:
            raise ValueError("Model file is truncated: no 'end of trees' marker")

        header: dict[str, str] = {}
        flags: set[str] = set()
        tree_blocks: list[dict[str, str]] = []
        current = header
        for line in lines:
            line = line.strip()
            if line == "end of trees":
                break
            if not line:
                continue
            key, separator, value = line.partition("=")
            if key == "Tree" and separator:
                current = {}
                tree_blocks.append(current)
            elif separator:
                current[key] = value
            elif current is header:
                flags.add(key)

        if int(header.get("num_class", "1")) != 1 or int(header.get("num_tree_per_iteration", "1")) != 1:
            raise UnsupportedModelError("Multi-class models are not supported")
        self._num_features = int(header["max_feature_idx"]) + 1

        objective = header.get("objective", "regression").split()
        self._sigmoid: float | None = None
        if objective[0] == "binary":
            self._sigmoid = 1.0
            for part in objective[1:]:
                name, _, value = part.partition(":")
                if name == "sigmoid":
                    self._sigmoid = float(value)
        elif objective[0] != "regression":
            raise UnsupportedModelError(f"Objective '{objective[0]}' is not supported")
        self._average_output = "average_output" in flags

        expected_trees = len(header["tree_sizes"].split()) if "tree_sizes" in header else len(tree_blocks)
        if len(tree_blocks) != expected_trees or not tree_blocks:
            raise ValueError(f"Model file holds {len(tree_blocks)} trees, header declares {expected_trees}")
        self._trees = [_Tree(i, block, self._num_features) for i, block in enumerate(tree_blocks)]

    @classmethod
    def from_file(cls, path: str | Path) -> "TextModelBooster":
        return cls(Path(path).read_text(encoding="utf-8"))

    def num_feature(self) -> int:
        return self._num_features

    def num_trees(self) -> int:
        return len(self._trees)

    def predict(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._num_features:
            raise ValueError(
                f"The number of features in data ({rows.shape[1]}) is not the same as it was "
                f"in training data ({self._num_features})"
            )
        if rows.shape[0] <= _SCALAR_ROW_LIMIT:
            raw = np.zeros(rows.shape[0], dtype=np.float64)
            for i, row in enumerate(rows.tolist()):
                total = 0.0
                for tree in self._trees:
                    total += tree.output_for_row(row)
                raw[i] = total
        else:
            raw = np.zeros(rows.shape[0], dtype=np.float64)
            for tree in self._trees:
                raw += tree.output_for_rows(rows)
        if self._average_output:
            raw /= len(self._trees)
        if self._sigmoid is None:
            return raw
        return 1.0 / (1.0 + np.exp(-self._sigmoid * raw))

    def save_model(self, path: str | Path) -> None:
        Path(path).write_text(self._model_text, encoding="utf-8", newline="\n")
