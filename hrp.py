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


def getRecBipart(closes, ddev, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2),
                                                      (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(closes, ddev, cItems0)
            cVar1 = getClusterVar(closes, ddev, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def getClusterVar(closes, ddev, cItems):
    # Compute variance per cluster

    closes = closes[closes.columns[cItems]]
    ret = np.log(closes / closes.shift()).fillna(0.0)
    quant = 1000000 * getIVP(ret.var(0).values) / closes.iloc[-1].values
    price = np.sum(closes * quant, 1)
    ret = np.log(price / price.shift()).fillna(0.0).values

    if ddev:
        ret[ret > 0] = 0.0
        cVar = np.power(ret, 2).sum() / len(ret)
    else:
        cVar = np.var(ret)

    return cVar


def getIVP(variance, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / variance
    ivp /= ivp.sum()
    return ivp


def cov_robust(X):
    oas = sklearn.covariance.OAS()
    oas.fit(X)
    return pd.DataFrame(oas.covariance_, index=X.columns, columns=X.columns)


def get_weights(closes, robust, ddev, cats, graph_path):
    ret = np.log(closes / closes.shift()).fillna(0.0)
    corr = cov2cor(cov_robust(ret) if robust else ret.cov())
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
    weights = getRecBipart(closes, ddev, clusters)
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


class WeightAdjust(Algo):

    def __init__(self, leverage=1.0, weight_round=2):
        super().__init__()
        self._leverage = leverage
        self._weight_round = weight_round

    def __call__(self, target):

        if 'weights' not in target.temp.keys():
            return True

        weights = pd.Series(target.temp['weights']).round(self.wround)
        if weights.sum().round(self.wround) > self.leverage:
            weights = ((self.leverage / weights.sum()).round(self.wround) * weights).round(self.wround)

            if weights.sum().round(self.wround) > self.leverage:
                weights[weights.idxmax()] += np.round(self.leverage - weights.sum(), self.wround)

        target.temp['weights'] = weights.to_dict()

        return True

    @property
    def leverage(self):
        return self._leverage

    @property
    def wround(self):
        return self._weight_round


class WeightsToPerm(Algo):

    def __call__(self, target):
        if 'weights' in target.temp.keys():
            target.perm['weights'] = target.temp['weights'].copy()

        return True


class WeightHRP(Algo):

    def __init__(self, plen, ptype, robust=False, ddev=False, cats=None, graph_path=None):
        super().__init__()

        self._robust = robust
        self._ddev = ddev
        self._plen = plen
        self._ptype = ptype
        self._cats = cats.copy()# if cats is not None else None
        self._grap_path = graph_path

    def __call__(self, target):
        selected = target.temp['selected']
        index = target.universe.index

        # match = re.compile('([0-9]+)([a-z])').match(self.period)
        # plen, ptype = int(match.group(1)), match.group(2)

        idx = target.now - relativedelta(**{self.ptype: self.plen})
        idx = index[np.abs(index - idx).argmin()]

        prices = target.universe[selected].loc[idx: target.now].copy()
        weights = get_weights(prices, self.robust, self.ddev, self._cats, self._grap_path)

        target.temp['weights'] = weights.to_dict()

        return True

    @property
    def robust(self):
        return self._robust

    @property
    def ddev(self):
        return self._ddev

    @property
    def plen(self):
        return self._plen

    @property
    def ptype(self):
        return self._ptype


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
            weights = self._adjust_weights(weights, np.round(1 - mw.sum().round(self.wround), self.wround),
                                     np.round(other.sum(), self.wround),
                                     self.gap, self.wround)
            mw[weights.index] = weights

            if np.round(mw.sum(), 0) != 1.0:
                mw[mw.idxmax()] += np.round(1 - mw.sum(), self.wround)

        target.temp['weights'] = mw.copy().to_dict()

        return True

    def _adjust_weights(self, weights, free, other, gap, wround):

        a = weights.copy()
        weights = weights.copy()

        n = 1
        while np.round(free / n, wround) > gap:
            n += 1
        n = np.min((np.max((n - 1, 1)), len(weights)))

        wta = np.round(free / n, wround)

        idx = weights.sort_values().index[:n] if free > 0 else weights.sort_values().index[-n:]
        for vix in idx:
            weights.loc[vix] += wta

        weights = weights.round(wround)

        if np.round(weights.sum() + other, 0) != 1:
            try:
                weights.loc[weights.idxmin()] += 1 - np.round(weights.sum() + other, wround)
            except Exception as e:
                traceback.print_tb(sys.exc_info()[2])

            weights = weights.round(wround)

            if np.round(weights.sum() + other, 0) != 1:
                return adjust_weights(weights, np.round(1 - weights.sum() - other, wround), other, gap, wround)

        return weights


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

        return True


class WeightTargetVol(Algo):

    def __init__(self, tvol, plen, ptype, kmax=1.0):
        super().__init__()

        self._tvol = tvol
        self._plen = plen
        self._ptype = ptype
        self._kmax = kmax

    def __call__(self, target):

        if 'weights' not in target.temp.keys():
            return True

        index = target.universe.index

        idx = target.now - relativedelta(**{self.ptype: self.plen})
        idx = index[np.abs(index - idx).argmin()]

        prices = target.universe[list(target.temp['weights'].keys())].loc[idx: target.now].copy()
        weights = pd.Series(target.temp['weights'])
        amounts = weights * 1000000 / prices.iloc[0]
        pvol = (prices * amounts).sum(1).to_returns().std() * np.sqrt(252)
        k = np.min((self.tvol / pvol, self.kmax))

        target.temp['weights'] = (weights * k).to_dict()

        return True

    @property
    def tvol(self):
        return self._tvol

    @property
    def plen(self):
        return self._plen

    @property
    def ptype(self):
        return self._ptype

    @property
    def kmax(self):
        return self._kmax


def min_max_rebalance(weights, minw, maxw, diff):

    nw = weights.copy()
    if diff < 0:
        diff = 0.0
        nw = weights[weights > minw]
        nw = nw - np.round(diff * nw / nw.sum(), 6)
        if np.any(nw < minw):
            diff = nw[nw < minw] - minw
            nw[nw < minw] = minw
    elif diff > 0:
        diff = 0.0
        nw = weights[weights < maxw]
        nw = nw + np.round(diff * nw / nw.sum(), 6)
        if np.any(nw > maxw):
            diff = nw[nw > maxw] - maxw
            nw[nw > maxw] = maxw

    weights.loc[nw.index] = nw
    if diff != 0:
        nw = min_max_rebalance(weights, minw, maxw, diff)
        weights.loc[nw.index] = nw

    return weights


class LimitWeights(Algo):

    """
    Modifies temp['weights'] based on weight limits.

    This is an Algo wrapper around ffn's limit_weights. The purpose of this
    Algo is to limit the weight of any one specifc asset. For example, some
    Algos will set some rather extreme weights that may not be acceptable.
    Therefore, we can use this Algo to limit the extreme weights. The excess
    weight is then redistributed to the other assets, proportionally to
    their current weights.

    See ffn's limit_weights for more information.

    Args:
        * limit (float): Weight limit.

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, min_limit=0.0, max_limit=1.0, leverage=1.0):
        super(LimitWeights, self).__init__()
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.leverage = leverage

    def __call__(self, target):
        if 'weights' not in target.temp:
            return True

        tw = target.temp['weights']
        if len(tw) == 0:
            return True

        minw = self.min_limit
        maxw = self.max_limit

        if isinstance(tw, dict):
            tw = pd.Series(tw)

        if not np.any(tw < minw) and not np.any(tw > maxw):
            return True

        tws = tw.sum().round(6)

        tw[tw < minw] = minw
        tw[tw > maxw] = maxw

        ntws = tw.sum().round(6)
        diff = np.round(tws - ntws, 8)

        tw = min_max_rebalance(tw, minw, maxw, diff)

        target.temp['weights'] = tw.to_dict()

        return True
