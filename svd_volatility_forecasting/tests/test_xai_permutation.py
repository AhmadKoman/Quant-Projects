import numpy as np

from evaluation import xai


def test_blocked_permutation_importance_identifies_signal_feature():
    rng = np.random.default_rng(0)
    T = 300
    X = rng.normal(size=(T, 3))
    # True variance depends strongly on feature 0
    y = 1.0 + (X[:, 0] ** 2)

    def predict_var(Xin: np.ndarray) -> np.ndarray:
        Xin = np.asarray(Xin)
        return 1.0 + (Xin[:, 0] ** 2)

    res = xai.blocked_permutation_importance_2d(
        X=X,
        y_true_var=y,
        feature_names=["f0", "f1", "f2"],
        predict_var=predict_var,
        loss="qlike",
        reps=10,
        block_len=10,
        seed=123,
    )
    # Importance for f0 should dominate
    imps = res.importances_mean
    assert imps[0] > imps[1]
    assert imps[0] > imps[2]

