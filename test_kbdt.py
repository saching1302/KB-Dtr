"""Tests for the guarantees claimed in the paper.  Run: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

import kbdt as K

KNOW, TABLE, CR = K.elicit()


def test_saaty_tfn_reciprocal():
    for k in range(1, 10):
        l, m, u = K.saaty_tfn(k)
        assert l <= m <= u
        assert np.allclose(K.saaty_tfn(1 / k), (1 / u, 1 / m, 1 / l))


def test_fuzzy_ahp_weights_valid_and_consistent():
    w = KNOW.global_weights
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    assert np.isclose(sum(KNOW.dim_weights), 1)
    assert all(v["max_expert"] < 0.10 for v in CR.values())       # Saaty's acceptance threshold


def test_fuzzy_ahp_recovers_consistent_matrix():
    true = np.array([0.5, 0.3, 0.2])
    _, crisp = K.fuzzy_ahp([np.outer(true, 1 / true)])
    assert np.argsort(crisp).tolist() == np.argsort(true).tolist()


def test_rule_base_is_monotone():
    fis = K.FuzzyReadinessInference(KNOW)
    for i, a in enumerate(fis.antecedents):
        for j, b in enumerate(fis.antecedents):
            if (a <= b).all():
                assert fis.consequents[i] <= fis.consequents[j]


def test_fkri_bounded_and_monotone_in_every_criterion():
    rng = np.random.default_rng(0)
    fis = K.FuzzyReadinessInference(KNOW)
    X = rng.random((2000, 9))
    base, _ = fis(X)
    assert ((base >= 0) & (base <= 1)).all()
    for j in range(9):
        X2 = X.copy()
        X2[:, j] = np.minimum(1, X2[:, j] + rng.random(2000) * 0.3)
        assert (fis(X2)[0] >= base - 1e-12).all()


def test_cap_rule_limits_readiness_without_top_management():
    fis = K.FuzzyReadinessInference(KNOW)
    X = np.ones((1, 9))
    X[0, K.CODES.index("TM")] = 0.0
    assert fis(X)[0][0] <= KNOW.cap_level + 1e-12


def test_simulation_is_reproducible_and_balanced():
    X1, y1 = K.simulate_enterprises(500, seed=3)
    X2, y2 = K.simulate_enterprises(500, seed=3)
    pd.testing.assert_frame_equal(X1, X2)
    assert (y1 == y2).all() and set(y1) == {0, 1, 2}


@pytest.fixture(scope="module")
def fitted():
    X, y = K.simulate_enterprises(400, seed=1)
    Xt, yt = K.simulate_enterprises(400, seed=2)
    return K.KBDTR(KNOW).fit(X, y), Xt, yt


def test_kbdtr_probabilities_and_skill(fitted):
    model, Xt, yt = fitted
    P = model.predict_proba(Xt)
    assert np.allclose(P.sum(axis=1), 1) and (P >= -1e-12).all()
    assert (model.predict(Xt) == yt).mean() > 0.6                 # chance is about 0.4
    assert 0 <= model.alpha_ <= 1


def test_kbdtr_never_contradicts_knowledge(fitted):
    model, Xt, _ = fitted
    assert K.monotonicity_violation_rate(model, Xt) == 0.0


def test_levers_are_non_negative(fitted):
    model, Xt, _ = fitted
    lev = K.improvement_levers(model.expected_level, Xt.iloc[[0]], top=9)
    assert (lev.gain >= -1e-12).all()


def test_perturbation_keeps_valid_weights():
    k = K.perturb(KNOW, 1.0, np.random.default_rng(0))
    assert np.isclose(k.global_weights.sum(), 1) and (k.global_weights > 0).all()
