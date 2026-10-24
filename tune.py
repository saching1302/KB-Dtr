"""Tune KB-DTR's boosting hyperparameters by cross-validation on training data only.

  python tune.py                 # tune on firms.csv (column 'readiness' = 0/1/2)
  python tune.py my_firms.csv    # tune on another file
  python tune.py --simulated     # tune on 1,600 simulated firms (the paper's benchmark)
"""
import sys

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import kbdt as K

know, _, _ = K.elicit()
if "--simulated" in sys.argv:
    X, y = K.simulate_enterprises(1600, seed=1000)
else:
    df = pd.read_csv(next((a for a in sys.argv[1:] if a.endswith(".csv")), "firms.csv"))
    X, y = df.drop(columns="readiness"), df["readiness"].to_numpy(int)

grid = {"learning_rate": [0.02, 0.05, 0.1], "max_iter": [150, 300, 600]}
search = GridSearchCV(K.KBDTR(know), grid, scoring="f1_macro",
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1).fit(X, y)
res = pd.DataFrame(search.cv_results_)[["param_learning_rate", "param_max_iter", "mean_test_score", "std_test_score"]]
print(res.sort_values("mean_test_score", ascending=False).round(4).to_string(index=False))
print("\nBest:", search.best_params_, "macro-F1 =", round(search.best_score_, 4))
print("Set these as defaults in KBDTR.__init__ (kbdt.py), then rerun run_experiments.py.")
