import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kick_analysis.linear import LinearClassifier


def make_data():
    """클래스마다 중심이 다른 점 60개 (특징 4개, 클래스 3개)"""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0, 0], [3, 0, 1, 0], [0, 3, 0, 1]], dtype=float)
    y = np.repeat([0, 1, 2], 20)
    x = centers[y] + rng.normal(scale=0.8, size=(60, 4))
    x[:, 3] *= 100  # 한 특징만 눈금이 크다. 표준화가 없으면 이 값이 결과를 좌우한다
    return x, y


def test_matches_scikit_learn():
    """직접 계산한 확률이 scikit-learn의 결과와 같아야 한다."""
    x, y = make_data()
    ours = LinearClassifier.fit(x, y, ["a", "b", "c"])
    reference = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x, y)

    np.testing.assert_allclose(ours.predict_proba(x), reference.predict_proba(x), atol=1e-8)
    assert (ours.predict(x) == reference.predict(x)).all()


def test_probabilities_sum_to_one():
    x, y = make_data()
    proba = LinearClassifier.fit(x, y, ["a", "b", "c"]).predict_proba(x)
    assert proba.shape == (60, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_save_load_roundtrip(tmp_path):
    x, y = make_data()
    model = LinearClassifier.fit(x, y, ["back", "front", "middle"])
    model.save(tmp_path / "weight.npz")

    loaded = LinearClassifier.load(tmp_path / "weight.npz")
    assert loaded.labels == ["back", "front", "middle"]
    np.testing.assert_allclose(loaded.predict_proba(x), model.predict_proba(x))
