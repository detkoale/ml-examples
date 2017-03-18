import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import scorer

data = load_boston()
X = data['data']
y = data['target']

X_scale = scale(X)

folds = KFold(n_folds=5, shuffle=True, random_state=42, n=len(X_scale))
pMean = list()

for p in np.linspace(1, 10, num=200):
    knRegressor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    knRegressor.fit(X=X_scale, y=y)
    scores = cross_val_score(knRegressor, X_scale, y,
                             cv=folds, scoring=scorer.mean_squared_error_scorer)
    score = scores.mean()
    print("P:", p, "score:", score)
    pMean.append(score)

maxScore = max(pMean)
bestP = [i for i, j in enumerate(pMean, 1) if j == maxScore]
print "Max p:", maxScore, "Recommend p:", bestP[0]
