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
from bt.algos import Rebalance
from matplotlib import pyplot as plt

import networkx as nx
from copy import deepcopy
from os.path import join
import sys
import traceback
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

#hrp

def cov2cor(X):
    try:
        D = np.zeros_like(X)
        d = np.sqrt(np.diag(X))
        np.fill_diagonal(D, d)
        DInv = np.linalg.inv(D)
        R = np.dot(np.dot(DInv, X), DInv)
        return pd.DataFrame(R, index=X.index, columns=X.columns)
    except Exception as e:
        traceback.print_tb(sys.exc_info()[2])


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


def get_weights(closes, robust, cats, graph_path):
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

    try:
        if cats is not None:
            ccats = cats[corr.columns].copy()

        widxed = weights.loc[corr.index]
        names = [s.replace(' ', '\n') for s in corr.columns]
        corr.index = names
        corr.columns = names
        corr = ((corr - corr.min()) / corr.max()).round(2)
        mst = nx.minimum_spanning_tree(nx.from_pandas_adjacency(corr, create_using=nx.MultiGraph()))

        legends = None
        if cats is not None:
            ccats.columns = names
            ccats = ccats.T
        else:
            ccats = pd.DataFrame({
                'Colors': list('b' * len(names)),
                'Shapes': list('o' * len(names))
            }, index=names)

        ccats['Sizes'] = pd.DataFrame({i.replace(' ', '\n'): w for i, w in zip(weights.index, weights)}, index=['Sizes']).T
        fs = np.min((20, len(weights)))
        fig = plt.figure(figsize=(fs + 5, fs + 2), dpi=80)
        cf = fig.add_subplot(111)
        draw_net(mst, ccats, cf)

        if cats is not None:
            leg = []
            sdict = dict()
            adict = dict()
            for row in ccats.iterrows():
                row = row[1]
                sdict[row.Strategy] = row.Colors
                adict[row.Asset] = row.Shapes
            for k, v in sdict.items():
                leg.append(mpatches.Patch(color=v, label=k))
            leg1 = fig.legend(handles=leg, title='Strategy', loc=7, fontsize='xx-large')
            # leg = []
            # for k, v in adict.items():
            #     leg.append(mlines.Line2D([], [], color='black', marker=v, linestyle='None',
            #               markersize=10, label=k))
            # leg2 = fig.legend(handles=leg, title='Asset', loc=1)

        fig.savefig(join(graph_path, str(closes.index[-1].date()) + '.png'))
        plt.close('all')
    except Exception as e:
        pass

    return weights


def draw_net(G, ccats, cf):

    try:
        p = cf.get_position().get_points()
        x0, y0 = p[0]
        x1, y1 = p[1]
        cf.set_position([x0 / 2, y0 / 2, x1 - x0, y1 - y0])
        cf.set_facecolor('w')

        pos = nx.drawing.spring_layout(G)  # default to spring layout

        for key in nx.drawing.spring_layout(G).keys():
            shape = ccats.loc[key]['Shapes']
            nx.draw_networkx_nodes(G, pos,
                                   alpha=0.75,
                                   linewidths=4,
                                   node_shape=shape,
                                   node_size=ccats.loc[key]['Sizes'] * 10000,
                                   node_color=ccats.loc[key]['Colors'],
                                   nodelist=[key])
        nx.draw_networkx_edges(G, pos, arrows=False, edge_color='grey', width=3.0)
        nx.draw_networkx_labels(G, pos)
        cf.set_axis_off()
    except Exception as e:
        traceback.print_tb(sys.exc_info()[2])


#hrp


def adjust_weights(weights, free, other, gap, round):

    a = weights.copy()
    weights = weights.copy()

    n = 1
    while np.round(free / n, round) > gap:
        n += 1
    n = np.min((np.max((n - 1, 1)), len(weights)))

    wta = np.round(free / n, round)

    idx = weights.sort_values().index[:n] if free > 0 else weights.sort_values().index[-n:]
    for vix in idx:
        weights.loc[vix] += wta

    weights = weights.round(round)

    if np.round(weights.sum() + other, 0) != 1:
        try:
            weights.loc[weights.idxmin()] += 1 - np.round(weights.sum() + other, round)
        except Exception as e:
            traceback.print_tb(sys.exc_info()[2])

        weights = weights.round(round)

        if np.round(weights.sum() + other, 0) != 1:
            return adjust_weights(weights, np.round(1 - weights.sum() - other, round), other, gap, round)

    return weights


class WeightHRP(Algo):

    def __init__(self, plen, ptype, robust=False, weight_round=2, cats=None, graph_path=None):
        super().__init__()

        self._robust = robust
        self._weight_round = weight_round
        self._plen = plen
        self._ptype = ptype
        self._cats = cats.copy()
        self._grap_path = graph_path

    def __call__(self, target):
        selected = target.temp['selected']
        index = target.universe.index

        # match = re.compile('([0-9]+)([a-z])').match(self.period)
        # plen, ptype = int(match.group(1)), match.group(2)

        idx = target.now - relativedelta(**{self.ptype: self.plen})
        idx = index[np.abs(index - idx).argmin()]

        prices = target.universe[selected].loc[idx: target.now].copy()
        weights = get_weights(prices, self.robust, self._cats, self._grap_path).round(self.wround)
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
        wgi = wd[np.where(wd < self.gap)[0]].index.values

        fw = np.abs(nw[wgi] - ow[wgi]).sum().round(self.wround)
        mw[wgi] = ow[wgi]

        if mw.sum().round(self.wround) != 1.0:
            weights = mw[mw != ow[mw.index]]
            if len(weights) == 0:
                weights = mw
            other = mw.loc[[i for i in mw.index if i not in weights.index]]
            weights = adjust_weights(weights, np.round(1 - mw.sum().round(self.wround), self.wround),
                                     np.round(other.sum(), self.wround),
                                     self.gap, self.wround)
            mw[weights.index] = weights

            if np.round(weights.sum(), 0) != 1.0:
                weights[weights.idxmax()] += np.round(1 - weights.sum(), self.wround)

            target.temp['weights'] = mw.copy().to_dict()
            target.perm['weights'] = mw.copy().to_dict()

        return True


class CheckFeeBankrupt(Algo):

    def __init__(self, fee_func):
        super().__init__()

        self._fee_func = fee_func

    def __call__(self, target):

        selected = target.temp.get('selected')
        weights = target.temp.get('weights')

        if selected is not None and weights is not None:

            strat = deepcopy(target)
            Rebalance()(strat)
            if np.any(strat.positions < 0):
                target.bankrupt = True
                target.temp.pop('weights', None)

            # weights = pd.Series(weights)
            # prices = target.universe[selected].loc[target.now].copy()
            #
            # fee = 0.0
            # for w, p in pd.concat((weights, prices[weights.index]), axis=1).values:
            #     fee += self._fee_func(target.value * np.abs(w) / p, p)
            #
            # target.bankrupt = np.round(fee / target.value, 2) >= .4 or target.value < 0
            # if target.bankrupt:
            #     target.temp.pop('weights', None)

        return True

