# -*- encode: utf-8 -*-

from datetime import timedelta
from dateutil.relativedelta import relativedelta
import re

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import linkage, dendrogram

from bt import Algo


#hrp

def cov2cor(X):
    D = np.zeros_like(X)
    d = np.sqrt(np.diag(X))
    np.fill_diagonal(D, d)
    DInv = np.linalg.inv(D)
    R = np.dot(np.dot(DInv, X), DInv)
    return pd.DataFrame(R, index=X.index, columns=X.columns)


def distance(corr):
    columns = corr.columns
    corr = np.sqrt((1 - corr) / 2.).values
    np.fill_diagonal(corr, 0.0)
    return pd.DataFrame(corr, index=columns, columns=columns)


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index

    return sortIx.tolist()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2),
                                                      (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def getClusterVar(cov, cItems):
    # Compute variance per cluster

    cov_ = cov.iloc[cItems, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def cov_robust(X):
    oas = sklearn.covariance.OAS()
    oas.fit(X)
    return pd.DataFrame(oas.covariance_, index=X.columns, columns=X.columns)


def get_weights(closes, robust):
    ret = np.log(closes / closes.shift()).fillna(0.0)
    corr = cov2cor(ret.cov() if robust else cov_robust(ret))
    dist = distance(corr)
    link = linkage(dist, 'ward')
    quasiIdx = dendrogram(link)['leaves']
    weights = getRecBipart(ret.cov(), quasiIdx)
    weights.index = closes.columns[weights.index]

    return weights.round(4)

#hrp


class WeightHRP(Algo):

    def __init__(self, plen, ptype, robust=False):
        super().__init__()

        self._robust = robust
        self._plen = plen
        self._ptype = ptype

    def __call__(self, target):
        selected = target.temp['selected']
        index = target.universe.index

        # match = re.compile('([0-9]+)([a-z])').match(self.period)
        # plen, ptype = int(match.group(1)), match.group(2)

        idx = target.now - relativedelta(**{self.ptype: self.plen})
        if self.ptype == 'weeks':
            idx = index[(index.year == idx.year) & (index.week == idx.week)][-1]
        elif self.ptype == 'months':
            idx = index[(index.year == idx.year) & (index.month == idx.month)][-1]
        elif self.ptype == 'years':
            idx = index[index.year == idx.year][-1]

        prices = target.universe[selected].loc[idx: target.now].copy()
        target.temp['weights'] = get_weights(prices, self.robust).to_dict()

        return True

    @property
    def robust(self):
        return self._robust

    @property
    def plen(self):
        return self._plen

    @property
    def ptype(self):
        return self._ptype

if __name__ == '__main__':
    pass