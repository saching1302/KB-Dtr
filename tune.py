from sklearn.model_selection import GridSearchCV, StratifiedKFold
import kbdt as K

know, _, _ = K.elicit()
X, y = K.simulate_enterprises(1600, seed=1000)   # or load your own CSV (step 6)
grid = {"learning_rate": [0.02, 0.05, 0.1], "max_iter": [150, 300, 600]}
search = GridSearchCV(K.KBDTR(know), grid, scoring="f1_macro",
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1).fit(X, y)
print(search.best_params_, round(search.best_score_, 4))
