"""KB-DTR: a knowledge-based AI decision-support model for predicting digital
transformation (DT) readiness from enterprise innovation data.

Pipeline (Fig. 1 of the paper)
  enterprise innovation data  9 TOE indicators in [0, 1] + firm context (size, sector)
  -> knowledge base           fuzzy pairwise judgements from several experts + linguistic rules
  -> fuzzy AHP (Buckley)      fuzzy and crisp criterion weights, consistency ratios
  -> fuzzy rule inference     Fuzzy Knowledge Readiness Index (FKRI), monotone by construction
  -> knowledge-constrained    ordinal gradient boosting on raw data + knowledge features,
     ordinal learner          with monotonicity constraints taken from the knowledge base
  -> decision support         readiness level, class probabilities and what-if improvement levers

Command line
  python kbdt.py knowledge                          # print criterion weights, consistency and the rule base
  python kbdt.py assess firms.csv [--out out.csv]   # score unlabelled firms with the knowledge model
  python kbdt.py evaluate firms.csv --label y       # repeated CV of every model on your labelled data
  python kbdt.py demo                               # decision-support report for 3 simulated firms

A firms.csv needs columns DI DA AA RD IO DS TM EC CP (scaled to [0, 1], blanks allowed),
size (0 = micro, 1 = small, 2 = medium, 3 = large) and sector (free text). For `evaluate`,
it also needs an ordinal label column (0 = Low, 1 = Medium, 2 = High).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# 1. Criteria: the TOE indicator set                                           #
# --------------------------------------------------------------------------- #
CRITERIA = [  # code, TOE dimension, criterion, operational indicator (scaled to [0, 1])
    ("DI", "Technology", "Digital infrastructure", "Share of core processes on integrated digital systems (ERP, cloud, IoT)"),
    ("DA", "Technology", "Data and analytics capability", "Extent of data-driven decision-making and analytics use"),
    ("AA", "Technology", "AI and automation adoption", "Share of business functions using AI or intelligent automation"),
    ("RD", "Organization", "R&D intensity", "R&D expenditure relative to turnover (normalised)"),
    ("IO", "Organization", "Innovation output", "Share of turnover from new or improved products and processes"),
    ("DS", "Organization", "Digital skills", "Share of employees with ICT skills or recent digital training"),
    ("TM", "Organization", "Top-management support", "Leadership commitment to an explicit DT strategy"),
    ("EC", "Environment", "Ecosystem collaboration", "Innovation co-operation with universities, start-ups and suppliers"),
    ("CP", "Environment", "Digital market pressure", "Pressure from digital competitors and customer expectations"),
]
CODES = [c[0] for c in CRITERIA]
NAMES = {c[0]: c[2] for c in CRITERIA}
DIMENSIONS = ["Technology", "Organization", "Environment"]
DIM_CODES = {d: [c[0] for c in CRITERIA if c[1] == d] for d in DIMENSIONS}
SECTORS = ["Manufacturing", "ICT", "Services", "Retail", "Construction"]
LEVELS = ["Low", "Medium", "High"]
TERMS = ["Very low", "Low", "Medium", "High", "Very high"]

# --------------------------------------------------------------------------- #
# 2. Knowledge base: fuzzy AHP over expert judgements                          #
# --------------------------------------------------------------------------- #
# Saaty-scale judgements (a, b): v  means "a is v times as important as b".
# Three judgement profiles synthesised from DT-readiness literature emulate a
# heterogeneous expert panel (technology-, organisation- and balance-oriented).
# Replace them with your own panel's judgements; nothing else changes.
REFERENCE_JUDGEMENTS = {
    "Technology-oriented": {
        "Dimensions": {("Technology", "Organization"): 2, ("Technology", "Environment"): 4, ("Organization", "Environment"): 3},
        "Technology": {("DI", "DA"): 1, ("DI", "AA"): 3, ("DA", "AA"): 3},
        "Organization": {("RD", "IO"): 1/2, ("RD", "DS"): 1/3, ("RD", "TM"): 1/3, ("IO", "DS"): 1/2, ("IO", "TM"): 1/2, ("DS", "TM"): 1},
        "Environment": {("EC", "CP"): 2},
    },
    "Organisation-oriented": {
        "Dimensions": {("Technology", "Organization"): 1/3, ("Technology", "Environment"): 2, ("Organization", "Environment"): 4},
        "Technology": {("DI", "DA"): 1/2, ("DI", "AA"): 2, ("DA", "AA"): 4},
        "Organization": {("RD", "IO"): 1, ("RD", "DS"): 1/3, ("RD", "TM"): 1/4, ("IO", "DS"): 1/2, ("IO", "TM"): 1/3, ("DS", "TM"): 1/2},
        "Environment": {("EC", "CP"): 1},
    },
    "Balanced": {
        "Dimensions": {("Technology", "Organization"): 1/2, ("Technology", "Environment"): 3, ("Organization", "Environment"): 3},
        "Technology": {("DI", "DA"): 1/2, ("DI", "AA"): 2, ("DA", "AA"): 3},
        "Organization": {("RD", "IO"): 1/2, ("RD", "DS"): 1/2, ("RD", "TM"): 1/3, ("IO", "DS"): 1, ("IO", "TM"): 1/2, ("DS", "TM"): 1},
        "Environment": {("EC", "CP"): 2},
    },
}
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def saaty_tfn(a: float) -> tuple[float, float, float]:
    """Crisp Saaty judgement (1..9 or a reciprocal) -> triangular fuzzy number (l, m, u)."""
    if a < 1:
        l, m, u = saaty_tfn(1 / a)
        return 1 / u, 1 / m, 1 / l
    k = int(round(a))
    return (1.0, 1.0, 1.0) if k == 1 else (float(max(1, k - 1)), float(k), float(min(9, k + 1)))


def pairwise_matrix(items: list[str], judgements: dict) -> np.ndarray:
    idx = {k: i for i, k in enumerate(items)}
    A = np.ones((len(items), len(items)))
    for (a, b), v in judgements.items():
        A[idx[a], idx[b]], A[idx[b], idx[a]] = v, 1 / v
    return A


def consistency_ratio(A: np.ndarray) -> float:
    n = len(A)
    if n < 3:
        return 0.0
    lam = np.max(np.linalg.eigvals(A).real)
    return float((lam - n) / (n - 1) / RANDOM_INDEX[n])


def fuzzy_ahp(crisp_matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Buckley's geometric-mean fuzzy AHP with geometric-mean aggregation of experts.

    Returns fuzzy weights (n, 3) as (l, m, u) and normalised centroid (crisp) weights (n,).
    """
    F = np.stack([np.array([[saaty_tfn(a) for a in row] for row in A]) for A in crisp_matrices])
    F = np.exp(np.log(F).mean(axis=0))            # aggregate experts: (n, n, 3)
    r = np.exp(np.log(F).mean(axis=1))            # fuzzy geometric mean of each row: (n, 3)
    s = r.sum(axis=0)
    w = np.column_stack([r[:, 0] / s[2], r[:, 1] / s[1], r[:, 2] / s[0]])
    crisp = w.mean(axis=1)
    return w, crisp / crisp.sum()


@dataclass(frozen=True)
class Knowledge:
    """Everything the fuzzy model needs: weights and the non-compensatory cap rule."""
    dim_weights: tuple            # crisp TOE dimension weights (3,)
    local_weights: tuple          # crisp within-dimension weights, one array per dimension
    cap_criterion: str = "TM"     # IF <cap_criterion> is Low THEN readiness is at most <cap_level>
    cap_level: float = 0.5
    synergy: float = 0.3          # weight of the T-O complementarity rule (socio-technical alignment)

    @property
    def global_weights(self) -> pd.Series:
        w = {c: self.dim_weights[i] * self.local_weights[i][j]
             for i, d in enumerate(DIMENSIONS) for j, c in enumerate(DIM_CODES[d])}
        return pd.Series(w)[CODES]


def elicit(judgements: dict = REFERENCE_JUDGEMENTS) -> tuple[Knowledge, pd.DataFrame, dict]:
    """Run fuzzy AHP on every block. Returns knowledge, Table 1 and consistency ratios."""
    blocks = {"Dimensions": DIMENSIONS, **DIM_CODES}
    fuzzy, crisp, cr = {}, {}, {}
    for block, items in blocks.items():
        mats = [pairwise_matrix(items, prof[block]) for prof in judgements.values()]
        fuzzy[block], crisp[block] = fuzzy_ahp(mats)
        agg = np.exp(np.log(np.stack(mats)).mean(axis=0))
        cr[block] = {"max_expert": max(consistency_ratio(A) for A in mats), "aggregated": consistency_ratio(agg)}
    know = Knowledge(tuple(crisp["Dimensions"]), tuple(crisp[d] for d in DIMENSIONS))
    rows = []
    for i, d in enumerate(DIMENSIONS):
        for j, c in enumerate(DIM_CODES[d]):
            lo, mid, up = fuzzy["Dimensions"][i] * fuzzy[d][j]     # TFN product (standard approximation)
            rows.append({"Dimension": d, "Code": c, "Criterion": NAMES[c],
                         "Indicator": next(x[3] for x in CRITERIA if x[0] == c),
                         "Fuzzy weight (l, m, u)": f"({lo:.3f}, {mid:.3f}, {up:.3f})",
                         "Crisp weight": know.dim_weights[i] * crisp[d][j]})
    table = pd.DataFrame(rows)
    table["Rank"] = table["Crisp weight"].rank(ascending=False).astype(int)
    return know, table, cr


def perturb(know: Knowledge, rho: float, rng: np.random.Generator) -> Knowledge:
    """Corrupt a share rho of the knowledge: blend weights with random ones and,
    with probability rho, attach the cap rule to a wrong criterion."""
    def mix(w):
        w = np.asarray(w)
        return (1 - rho) * w + rho * rng.dirichlet(np.ones(len(w)))
    cap = know.cap_criterion
    if rng.random() < rho:
        cap = rng.choice([c for c in CODES if c != know.cap_criterion])
    return replace(know, dim_weights=tuple(mix(know.dim_weights)),
                   local_weights=tuple(mix(w) for w in know.local_weights), cap_criterion=str(cap))


# --------------------------------------------------------------------------- #
# 3. Fuzzy rule inference -> Fuzzy Knowledge Readiness Index (FKRI)            #
# --------------------------------------------------------------------------- #
def ruspini(x: np.ndarray) -> np.ndarray:
    """Low / Medium / High triangular memberships on [0, 1] that sum to one."""
    low, high = np.clip(1 - 2 * x, 0, 1), np.clip(2 * x - 1, 0, 1)
    return np.column_stack([low, 1 - low - high, high])


class FuzzyReadinessInference:
    """27-rule TOE rule base with product t-norm and centre-average defuzzification.

    Rule (a, b, c): IF Technology is L_a AND Organization is L_b AND Environment is L_c
    THEN readiness is T_k with k = round(2 * [(1 - s) * sum_d w_d * level_d + s * min(level_T, level_O)])
    in {0..4} (5 output terms). The min term encodes complementarity: technology pays off only
    with matching organisational capability. Consequents never decrease in any antecedent level,
    and Ruspini partitions turn the rule base into multilinear interpolation, so the FKRI is
    monotone in every criterion. A cap rule adds non-compensatory knowledge:
    IF top-management support is Low THEN readiness is at most Medium.
    """

    def __init__(self, know: Knowledge):
        self.know = know
        self.antecedents = np.array(list(product(range(3), repeat=3)))
        a, sy = self.antecedents, know.synergy
        level = (1 - sy) * a @ np.asarray(know.dim_weights) + sy * np.minimum(a[:, 0], a[:, 1])
        self.consequents = np.rint(2 * level).astype(int)
        self.centres = self.consequents / 4

    def dimension_scores(self, Xc: np.ndarray) -> np.ndarray:
        cols = {c: i for i, c in enumerate(CODES)}
        return np.column_stack([Xc[:, [cols[c] for c in DIM_CODES[d]]] @ np.asarray(self.know.local_weights[i])
                                for i, d in enumerate(DIMENSIONS)])

    def __call__(self, Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = self.dimension_scores(Xc)
        mu = [ruspini(D[:, d]) for d in range(3)]
        a = self.antecedents
        firing = mu[0][:, a[:, 0]] * mu[1][:, a[:, 1]] * mu[2][:, a[:, 2]]     # (N, 27), rows sum to 1
        fkri = firing @ self.centres
        k = self.know
        cap = k.cap_level + (1 - k.cap_level) * (1 - ruspini(Xc[:, CODES.index(k.cap_criterion)])[:, 0])
        return np.minimum(fkri, cap), D

    def rule_table(self) -> pd.DataFrame:
        return pd.DataFrame({d: [LEVELS[i] for i in self.antecedents[:, j]] for j, d in enumerate(DIMENSIONS)}
                            | {"Readiness": [TERMS[k] for k in self.consequents]})


# --------------------------------------------------------------------------- #
# 4. Models                                                                    #
# --------------------------------------------------------------------------- #
def _criteria(X: pd.DataFrame) -> np.ndarray:
    return X[CODES].to_numpy(float)


def _impute(Xc: np.ndarray, med: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(Xc), med, Xc)


def _sector_code(X: pd.DataFrame) -> np.ndarray:
    return X["sector"].map({s: i for i, s in enumerate(SECTORS)}).to_numpy(float)


class _Constant:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, Z):
        return np.column_stack([np.full(len(Z), 1 - self.p), np.full(len(Z), self.p)])


def _ordinal_fit(make, Z, y):
    """Frank & Hall ordinal decomposition: one binary model per threshold y > k."""
    out = []
    for k in (0, 1):
        t = (y > k).astype(int)
        out.append(make().fit(Z, t) if 0 < t.sum() < len(t) else _Constant(float(t.mean())))
    return out


def _ordinal_proba(models, Z):
    a = models[0].predict_proba(Z)[:, 1]
    b = np.minimum(models[1].predict_proba(Z)[:, 1], a)    # keeps P(y>1) <= P(y>0)
    return np.column_stack([1 - a, a - b, b])


class _Base(BaseEstimator, ClassifierMixin):
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def expected_level(self, X):
        return self.predict_proba(X) @ np.arange(3)


class KnowledgeOnly(_Base):
    """Pure knowledge model. Data only calibrate the score into class probabilities
    (two 1-D logistic thresholds); no criterion weight is learned from data."""

    def __init__(self, know: Knowledge, rule_inference: bool = True):
        self.know, self.rule_inference = know, rule_inference

    def score(self, X):
        Xc = _impute(_criteria(X), self.medians_)
        return self.fis_(Xc)[0] if self.rule_inference else Xc @ self.know.global_weights.to_numpy()

    def fit(self, X, y):
        self.medians_ = np.nan_to_num(np.nanmedian(_criteria(X), axis=0), nan=0.5)
        self.fis_ = FuzzyReadinessInference(self.know)
        self.cal_ = _ordinal_fit(lambda: LogisticRegression(C=100.0), self.score(X)[:, None], np.asarray(y))
        self.classes_ = np.arange(3)
        return self

    def predict_proba(self, X):
        return _ordinal_proba(self.cal_, self.score(X)[:, None])


class DataDriven(_Base):
    """Data-only baselines: multinomial logistic regression or random forest."""

    def __init__(self, kind: str = "lr", random_state: int = 0):
        self.kind, self.random_state = kind, random_state

    def fit(self, X, y):
        num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        prep = ColumnTransformer([("num", num, CODES + ["size"]),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), ["sector"])])
        clf = (LogisticRegression(C=1.0, max_iter=2000) if self.kind == "lr" else
               RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=self.random_state))
        self.model_ = make_pipeline(prep, clf).fit(X, y)
        self.classes_ = np.arange(3)
        return self

    def predict_proba(self, X):
        P = np.zeros((len(X), 3))
        P[:, self.model_.classes_] = self.model_.predict_proba(X)
        return P


class KBDTR(_Base):
    """Knowledge-constrained ordinal gradient boosting (the proposed model).

    kb_features: add the FKRI and the three TOE dimension scores as inputs.
    monotone:    constrain every capability input (and knowledge feature) to have a
                 non-decreasing effect, as the knowledge base asserts.
    arbitration: blend the learner with the calibrated knowledge model, P = a*P_KB + (1-a)*P_GBM,
                 with a chosen by inner cross-validation (ranked probability score). Small or
                 noisy samples push a up (trust the experts); large samples push it down.
    With all three off, it is a plain ordinal GBM (the data-only reference).
    """

    def __init__(self, know: Knowledge | None = None, kb_features: bool = True, monotone: bool = True,
                 arbitration: bool = True, max_iter: int = 300, learning_rate: float = 0.05,
                 random_state: int = 0):
        self.know, self.kb_features, self.monotone, self.arbitration = know, kb_features, monotone, arbitration
        self.max_iter, self.learning_rate, self.random_state = max_iter, learning_rate, random_state

    def _design(self, X):
        Xc = _criteria(X)
        parts = [Xc, X[["size"]].to_numpy(float), _sector_code(X)[:, None]]
        if self.kb_features:
            fkri, D = self.fis_(_impute(Xc, self.medians_))
            parts += [D, fkri[:, None]]
        return np.hstack(parts)

    def fit(self, X, y):
        self.medians_ = np.nan_to_num(np.nanmedian(_criteria(X), axis=0), nan=0.5)
        self.fis_ = FuzzyReadinessInference(self.know) if self.kb_features else None
        n_kb = 4 if self.kb_features else 0
        cst = ([1] * len(CODES) + [0, 0] + [1] * n_kb) if self.monotone else None
        cat = np.array([False] * len(CODES) + [False, True] + [False] * n_kb)
        make = lambda: HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate, max_leaf_nodes=8, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=False, monotonic_cst=cst, categorical_features=cat,
            random_state=self.random_state)
        y = np.asarray(y)
        self.models_ = _ordinal_fit(make, self._design(X), y)
        self.classes_ = np.arange(3)
        self.alpha_ = 0.0
        if self.arbitration:
            self.prior_ = KnowledgeOnly(self.know).fit(X, y)
            folds = int(min(5, np.bincount(y, minlength=3).min()))
            if folds >= 2:
                Pg, Pk = np.zeros((len(y), 3)), np.zeros((len(y), 3))
                for tr, te in StratifiedKFold(folds, shuffle=True, random_state=self.random_state).split(X, y):
                    sub = dict(self.get_params(), arbitration=False)
                    Pg[te] = KBDTR(**sub).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
                    Pk[te] = KnowledgeOnly(self.know).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
                grid = np.linspace(0, 1, 21)
                self.alpha_ = float(grid[np.argmin([ranked_probability_score(y, g * Pk + (1 - g) * Pg) for g in grid])])
        return self

    def predict_proba(self, X):
        P = _ordinal_proba(self.models_, self._design(X))
        return P if self.alpha_ == 0 else self.alpha_ * self.prior_.predict_proba(X) + (1 - self.alpha_) * P


# --------------------------------------------------------------------------- #
# 5. Evaluation and decision support                                           #
# --------------------------------------------------------------------------- #
def ranked_probability_score(y, P) -> float:
    """Proper scoring rule for ordinal classes (lower is better)."""
    cum_p = np.cumsum(P, axis=1)[:, :-1]
    cum_y = (np.asarray(y)[:, None] <= np.arange(P.shape[1] - 1)).astype(float)
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1)))


def monotonicity_violation_rate(model, X: pd.DataFrame, step: float = 0.1) -> float:
    """Share of (firm, criterion) pairs where raising the criterion by `step`
    lowers the expected readiness level, i.e. predictions contradicting the knowledge base."""
    base = model.expected_level(X)
    viol = tot = 0
    for c in CODES:
        m = (X[c].notna() & (X[c] < 1)).to_numpy()
        X2 = X.copy()
        X2.loc[m, c] = np.minimum(1.0, X2.loc[m, c] + step)
        viol += int(np.sum(model.expected_level(X2)[m] < base[m] - 1e-9))
        tot += int(m.sum())
    return viol / max(tot, 1)


def evaluate(model, X_train, y_train, X_test, y_test, mvr: bool = True) -> dict:
    model.fit(X_train, y_train)
    P = model.predict_proba(X_test)
    pred = P.argmax(axis=1)
    out = {"macro_f1": f1_score(y_test, pred, average="macro"),
           "qwk": cohen_kappa_score(y_test, pred, weights="quadratic"),
           "auc": roc_auc_score(y_test, np.clip(P, 1e-12, 1), multi_class="ovr", average="macro")}
    if mvr:
        out["mvr"] = monotonicity_violation_rate(model, X_test)
    return out


def improvement_levers(score_fn, firm: pd.DataFrame, step: float = 0.1, top: int = 3) -> pd.DataFrame:
    """What-if analysis: readiness gain from improving each criterion by `step` (one firm)."""
    base = float(score_fn(firm)[0])
    rows = []
    for c in CODES:
        v = firm[c].iloc[0]
        if pd.isna(v) or v >= 1:
            continue
        f = firm.copy()
        f[c] = min(1.0, v + step)
        rows.append({"code": c, "criterion": NAMES[c], "current": v, "gain": float(score_fn(f)[0]) - base})
    return pd.DataFrame(rows).sort_values("gain", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Controlled benchmark: simulated enterprise innovation data                #
# --------------------------------------------------------------------------- #
# The data-generating process is NOT the knowledge base. Its true weights differ
# from the elicited ones (e.g. experts under-rate ecosystem collaboration), and it
# contains capability complementarities that no rule encodes. Only the top-management
# bottleneck and the positive direction of every effect agree with expert knowledge.
TRUE_BETA = dict(DI=0.10, DA=0.17, AA=0.07, RD=0.06, IO=0.11, DS=0.14, TM=0.14, EC=0.14, CP=0.07)
LOADING = dict(DI=0.7, DA=0.7, AA=0.6, RD=0.5, IO=0.6, DS=0.6, TM=0.4, EC=0.5, CP=0.3)
SECTOR_CAPABILITY = dict(Manufacturing=0.0, ICT=0.5, Services=0.1, Retail=-0.1, Construction=-0.3)
SECTOR_SHIFT = dict(Manufacturing=0.0, ICT=0.05, Services=0.02, Retail=-0.02, Construction=-0.04)


def _latent_readiness(n: int, rng: np.random.Generator):
    sector = rng.choice(SECTORS, size=n, p=[0.30, 0.15, 0.25, 0.20, 0.10])
    size = rng.choice(4, size=n, p=[0.35, 0.35, 0.20, 0.10])
    g = rng.normal(0.3 * (size - 1) + pd.Series(sector).map(SECTOR_CAPABILITY).to_numpy(), 1.0)
    x = {c: norm.cdf(LOADING[c] * g + np.sqrt(1 - LOADING[c] ** 2) * rng.standard_normal(n)) for c in CODES}
    r = (sum(TRUE_BETA[c] * x[c] for c in CODES)
         + 0.30 * x["DA"] * x["DS"]                    # analytics pays off only with skilled staff
         + 0.15 * x["DI"] * x["AA"]                    # automation needs infrastructure
         - 0.15 * (x["TM"] < 0.25)                     # organisational bottleneck
         + pd.Series(sector).map(SECTOR_SHIFT).to_numpy()
         + rng.normal(0, 0.05, n))
    return x, r, size, sector


@lru_cache(maxsize=1)
def _cutoffs() -> tuple[float, float]:
    _, r, _, _ = _latent_readiness(200_000, np.random.default_rng(12345))
    return tuple(np.quantile(r, [0.40, 0.75]))     # 40 % Low, 35 % Medium, 25 % High


def simulate_enterprises(n: int, seed: int, noise: float = 0.05, missing: float = 0.05):
    """Simulated enterprise innovation survey: features X and ordinal readiness y (0/1/2)."""
    rng = np.random.default_rng(seed)
    x, r, size, sector = _latent_readiness(n, rng)
    X = pd.DataFrame({c: np.clip(x[c] + rng.normal(0, noise, n), 0, 1) for c in CODES})
    X = X.mask(rng.random(X.shape) < missing)
    X["size"], X["sector"] = size, sector
    return X, np.digitize(r, _cutoffs())


# --------------------------------------------------------------------------- #
# 7. Command line                                                              #
# --------------------------------------------------------------------------- #
def _report(firm, model, fis, med, idx):
    Xc = _impute(_criteria(firm), med)
    fkri, D = fis(Xc)
    P = model.predict_proba(firm)[0]
    lev = improvement_levers(model.expected_level, firm)
    print(f"Firm {idx}: {firm['sector'].iloc[0]}, size class {int(firm['size'].iloc[0])}")
    print(f"  TOE scores  T={D[0, 0]:.2f}  O={D[0, 1]:.2f}  E={D[0, 2]:.2f}   FKRI={fkri[0]:.2f}")
    print(f"  Readiness   {LEVELS[int(P.argmax())]}  (P Low/Med/High = {P[0]:.2f}/{P[1]:.2f}/{P[2]:.2f})")
    print("  Levers      " + "; ".join(f"{r.criterion} +0.1 -> +{r.gain:.3f} levels" for r in lev.itertuples()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["knowledge", "assess", "evaluate", "demo"])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--label", default="readiness")
    ap.add_argument("--out")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    know, table, cr = elicit()
    fis = FuzzyReadinessInference(know)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)

    if a.command == "knowledge":
        print(table.drop(columns="Indicator").to_string(index=False, float_format="%.3f"))
        print("\nConsistency ratios:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in cr.items()})
        print("\nRule base (27 rules + cap rule: IF TM is Low THEN readiness <= Medium)")
        print(fis.rule_table().to_string())
    elif a.command == "assess":
        X = pd.read_csv(a.csv)
        Xc = _impute(_criteria(X), np.nan_to_num(np.nanmedian(_criteria(X), axis=0), nan=0.5))
        fkri, D = fis(Xc)
        out = X.assign(T_score=D[:, 0], O_score=D[:, 1], E_score=D[:, 2], FKRI=fkri,
                       level=[LEVELS[i] for i in ruspini(fkri).argmax(axis=1)])
        out["top_lever"] = [improvement_levers(lambda f: fis(_impute(_criteria(f), Xc[i]))[0], X.iloc[[i]], top=1)
                            ["criterion"].iloc[0] for i in range(len(X))]
        out.to_csv(a.out, index=False) if a.out else print(out.to_string(index=False, float_format="%.3f"))
    elif a.command == "evaluate":
        from sklearn.model_selection import RepeatedStratifiedKFold
        df = pd.read_csv(a.csv)
        X, y = df.drop(columns=a.label), df[a.label].to_numpy(int)
        models = {"FKRI (knowledge only)": lambda: KnowledgeOnly(know), "Logistic regression": lambda: DataDriven("lr"),
                  "Random forest": lambda: DataDriven("rf"), "Ordinal GBM": lambda: KBDTR(know, False, False, False),
                  "KB-DTR": lambda: KBDTR(know)}
        res = []
        for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=0).split(X, y)):
            for name, make in models.items():
                res.append({"model": name, **evaluate(make(), X.iloc[tr], y[tr], X.iloc[te], y[te])})
        print(pd.DataFrame(res).groupby("model", sort=False).agg(["mean", "std"]).round(3).to_string())
    else:
        X, y = simulate_enterprises(1600, seed=1)
        model = KBDTR(know).fit(X, y)
        Xt, _ = simulate_enterprises(3, seed=99)
        for i in range(3):
            _report(Xt.iloc[[i]], model, fis, model.medians_, i + 1)


if __name__ == "__main__":
    main()
