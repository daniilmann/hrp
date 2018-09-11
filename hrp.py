# -*- encode: utf-8 -*-

from datetime import timedelta
from dateutil.relativedelta import relativedelta
import re

import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import sklearn
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from bt import Algo
from matplotlib import pyplot as plt


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
    quasiIdx = np.array(dendrogram(link)['leaves'])
    clusters = quasiIdx
    # acceleration = np.diff(link[:, 2], 2)[::-1]
    # # ck = np.where(acceleration >= np.mean(acceleration))[0][-1] + 2
    # ck = acceleration.argmax() + 2
    # cluster_idx = fcluster(link, ck, criterion='maxclust') - 1
    # clusters = pd.Series()
    # cidx = []
    # for cn in np.unique(cluster_idx):
    #     idx = np.where(cluster_idx == cn)[0]
    #     cidx = np.where(cluster_idx ==cn)[0][0]
    #     clusters.loc[cidx] = quasiIdx[idx]
    # clusters = clusters.sort_index().values
    weights = getRecBipart(ret.cov(), clusters)
    weights.index = closes.columns[weights.index]

    return weights

#hrp


def adjust_weights(weights, free, other, gap, round):

    weights = weights.copy()

    n = 1
    while np.round(free / n, round) > gap:
        n += 1
    n = np.max((n - 1, 1))

    wta = np.round(free / n, round)

    for vix in weights.sort_values().index[:n]:
        weights.loc[vix] += wta

    weights = weights.round(round)

    if np.round(weights.sum() + other, 0) != 1:
        weights.loc[weights.idxmin()] += 1 - np.round(weights.sum() + other, round)

        weights = weights.round(round)

        if np.round(weights.sum() + other, 0) != 1:
            return adjust_weights(weights, np.round(1 - weights.sum() - other, round), other, gap, round)

    return weights


class WeightHRP(Algo):

    def __init__(self, plen, ptype, robust=False, weight_round=2):
        super().__init__()

        self._robust = robust
        self._weight_round = weight_round
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
        weights = get_weights(prices, self.robust).round(self.wround)
        if weights.sum().round(2) != 1.0:
            weights = adjust_weights(weights, np.round(1 - weights.sum(), self.wround), 0, 0, self.wround)
        target.temp['weights'] = weights.to_dict()
        target.perm['weights'] = weights.to_dict()

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

    @property
    def wround(self):
        return self._weight_round

class SaveWeights(Algo):

    def __call__(self, target):

        if 'weights' not in target.perm.keys():
            return True

        target.temp['old_weights'] = target.perm['weights'].copy()

        return True


class GapWeights(object):

    def __init__(self, gap, weight_round=2):
        # super().__init__()

        self._gap = gap
        self._wround = weight_round

    @property
    def gap(self):
        return self._gap

    @property
    def wround(self):
        return self._wround

    def __call__(self, target):

        if 'old_weights' not in target.temp.keys():
            return True

        ow = pd.Series(target.temp['old_weights'])
        nw = pd.Series(target.temp['weights'])
        mw = pd.Series(target.temp['weights'].copy())

        wd = np.abs(nw - ow).round(self.wround)
        wd = wd[wd != 0.0]
        wgi = mw.index.values[np.where(wd < self.gap)[0]]

        fw = 0.0
        for wix in wgi:
            fw += np.round(np.abs(nw[wix] - ow[wix]), 2)
            mw[wix] = ow[wix]

        if fw != 0.0:
            weights = mw.loc[[i for i in mw.index if i not in wgi]]
            weights = adjust_weights(weights, fw, mw.loc[wgi].sum(), self.gap, self.wround)
            mw[weights.index] = weights

            target.temp['weights'] = mw.copy().to_dict()
            target.perm['weights'] = mw.copy().to_dict()

        return True
