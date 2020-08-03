#!python
# !/usr/bin/env python
from __future__ import print_function
import datetime
import numpy as np
import warnings
import time
from tqdm import tqdm
from scipy.signal import savgol_filter
import numba
from numba.decorators import jit
from joblib import Parallel, delayed, parallel_backend
import logging
from copy import deepcopy

logger = logging.getLogger()

warnings.filterwarnings("ignore")

###---------------------------------- Built in Function --------------------------------------###

def twDTW_spring_subset_wise(uncls_feaIms, tsList, cropROI_mn, tsTrain, par_DTW, daysInterval, K_value):
    [d1, d2] = uncls_feaIms.shape[:2]
    clsIm_mn = np.zeros((d1, d2))  # Classified image
    distIm_mn = np.zeros((d1, d2))  # Minimum distance image
    Y_doy = datetime2doy_v2_jit(tsList, daysInterval)
    for i in tqdm(range(d2)):
        tstart = time.time()
        [pre_lb_col, dist_min_col] = twDTWS_col_wise(i, uncls_feaIms, cropROI_mn, Y_doy, tsTrain, par_DTW, K_value)
        clsIm_mn[:, i] = pre_lb_col.astype(int)
        distIm_mn[:, i] = dist_min_col
        #print("The elapsed time for applying the twDTW_spring with crop mapping in the time series subset image: {0:.2f} second".format(time.time() - tstart))
    return clsIm_mn, distIm_mn


def datetime2doy(dt):
    """Set data with the select interval dates"""
    a = dt[:]
    a2 = a[:]
    a2[:] = [(datetime.datetime(a[x].year, 1, 1) - datetime.timedelta(days=1)) for x in range(len(a))]
    a[:] = [(a[x] - a2[x]).days for x in range(len(a))]
    return a

def datetime2doy_v2(dt, daysInterval):
    dt_list = [None] * len(dt)
    delta = dt[0] - (datetime.datetime(dt[0].year, 1, 1) - datetime.timedelta(days=1))
    dt_list[:] = [((delta).days + (daysInterval * x)) for x in range(len(dt))]
    return dt_list
datetime2doy_v2_jit = numba.jit(datetime2doy_v2)


def twDTWS_col_wise(col_id, uncls_feaIms, cropROI_mn, Y_doy, tsTrain, par_DTW, K_value):
    """Reorganized Dataset, Scale dataset, and ROI Efficiency"""
    rows = uncls_feaIms.shape[0]
    pre_lb_col = np.zeros((rows, 1))
    dist_min_col = np.zeros((rows, 1))
    ts_fea = np.transpose(np.squeeze(uncls_feaIms[:, col_id, :, :]), (2, 1, 0))
    ts_DoY = np.tile((Y_doy), (rows, 1))  # in matlab: ts_DoY = repmat(Y_doy,1,rows);
    ts_fea_scale = np.copy(ts_fea)
    for i in range(ts_fea.shape[2]):
        ts_fea_scale[:, :, i] = scaleData(ts_fea[:, :, i])

    roi_ix, = np.where(cropROI_mn[:, col_id] > 0)
    tsTest = {"fea": ts_fea_scale[:, :, roi_ix],
              "DoY": ts_DoY[roi_ix, :]}

    # ---------------Classification
    # tstart = time.time()
    [pre_lb, dist_min] = KNN_twDTWS(K_value, tsTrain, tsTest, par_DTW)
    pre_lb_col[:, 0][roi_ix] = pre_lb  # pre_lb_col(roi_ix) = pre_lb;
    dist_min_col[:, 0][roi_ix] = dist_min
    # print("Elapsed: %f detik" % ((time.time() - tstart)))
    return pre_lb_col[:, 0].astype(int), dist_min_col[:, 0]


def scaleData(data):
    minimums = data.min(axis=0)
    ranges = (data.max(axis=0) - minimums)
    # in matlab: scale_data = (data - repmat(minimums, size(data, 1), 1)) ./ repmat(ranges, size(data, 1), 1);
    # scale_data = (data - np.tile((minimums), (len(data),1)))/ np.tile((ranges), (len(data),1))
    scale_data = np.divide((data - np.tile((minimums), (len(data), 1))), np.tile((ranges), (len(data), 1)))
    return scale_data


def KNN_twDTWS(k, tsTrain, tsTest, par_DTW):
    """Reorganized Dataset, Scale dataset, and ROI Efficiency
    KNN classification based on twDTWS distance
    Data normalization, and feature selection should be performed beforehand.
    We can also produce a posterior probability.
    tsTrain: refers to query sequence
    tsTest: refers to database sequence
    """

    # 1. Retrieve variables

    # Train

    tr_lb = tsTrain["lb"][0]  # 1D List: 1,2,3,... Has to be this format, otherwise unforeseen errors will occur.
    tr_fea = tsTrain["fea"][0]  # 3D numpy array, 1-3 dims: Time series, Features, Observations
    tr_DoY = tsTrain["DoY"][0]  # 2D numpy array, 1-2 dims: Time series, Observations
    num_tr = len(tr_lb)
    num_cls = max(tr_lb)

    # Test

    ts_fea = tsTest["fea"]
    ts_DoY = tsTest["DoY"]
    num_ts = tsTest["fea"].shape[2]

    # Params

    alpha = par_DTW["alpha"][0][0][0]  # par_DTW.alpha;
    theta = par_DTW["theta"][0][0][0]  # par_DTW.theta;
    beta = par_DTW["beta"][0][0][0]  # par_DTW.beta;
    epsilon = par_DTW["epsilon"][0][0][0]  # par_DTW.epsilon;

    # 2. KNN process

    dist_min = np.zeros((num_ts, 1))
    if k > num_tr:
        k = num_tr
    cls_temp = np.tile(range(1, (num_cls[0] + 1)), (k))

    ### Parallelization of twDTW_Spring_li_v2 using joblib :
    # with parallel_backend("loky", inner_max_num_threads=2):
    with parallel_backend('threading', n_jobs=32):
        the_vals = Parallel(n_jobs=32, max_nbytes=None)(delayed(par_KNN_twDTWS)(i, ts_fea, ts_DoY, tr_fea, tr_DoY,
                                                           alpha, beta, theta, epsilon, dist_min,
                                                           cls_temp, num_tr, k, tr_lb) for i in range(0, num_ts))
    a, b = zip(*the_vals);
    # out={"dist_min": b, "pre_lb":a};
    return a, b


def par_KNN_twDTWS(i, ts_fea, ts_DoY, tr_fea, tr_DoY, alpha, beta, theta, epsilon, dist_min, cls_temp, num_tr, k, tr_lb):
    """Parallelization of KNN_twDTWS"""
    pth = ""
    the_path = np.frombuffer(pth, dtype="uint8")
    dist_i = np.zeros((num_tr, 1))
    for j in range(num_tr):
        queryX = tr_fea[:, :, j][~np.isnan(tr_DoY[:, j])]
        doyX = tr_DoY[:, j][~np.isnan(tr_DoY[:, j])]
        temp_testY = ts_fea[:, :, i]
        # make writeable, to prevent error (ValueError: assignment destination is read-only)
        testY = deepcopy(temp_testY)
        testY[np.isnan(testY)] = 0
        doyY = ts_DoY[i, :]
        alpha, beta, theta, epsilon = alpha, beta, theta, epsilon
        par_unique = 1
        par_path = the_path
        len_tr_DoY = len(tr_DoY[:, j][~np.isnan(tr_DoY[:, j])])
        twDTW = twDTW_Spring_li_v2_numba(queryX, doyX, testY, doyY, alpha, beta, theta, epsilon, par_unique, par_path)
        dist_i[j, 0] = twDTW / len_tr_DoY

    dist_i_sort = np.sort(dist_i, axis=None)  # sort(dist_i)
    ix = np.argsort(dist_i, axis=None)
    if k > 1:
        return np.argmax(np.sum((tr_lb[ix[0:k]] == cls_temp).astype(int), axis=0)) + 1, dist_min[i, 0]
    else:
        return tr_lb[ix[0]][0], dist_i_sort[0]


@jit(nopython=True)
def twDTW_Spring_li_v2_numba(queryX, doyX, testY, doyY, alpha, beta, theta, epsilon, par_unique, par_path):
    """twDTW_Spring using numba for optimization time execution"""
    #     if(isinstance(queryX, list) and isinstance(testY, list)):
    #         queryX = queryX[:]
    #         testY =  testY[:]
    #     elif(isinstance(queryX, np.ndarray) and isinstance(testY, np.ndarray)):
    #         N, fNX = queryX.shape
    #         M, fMX = testY.shape
    #         if N < fNX or M < fMX:
    #             print("Make sure the signals are arranged column-wisely\n")
    #         if fNX != fMX:
    #             print("ERRORR !..Two inputs have different dimensions\n")
    #         if N > M:
    #              print("ERRORR !..Query sequence should not be longer than the test sequence\n")
    #     else:
    #         print("Only support vector or matrix inputs\n")
    # 2. SPRING DTW
    # 2.1 Define return variables
    N, fNX = queryX.shape
    M, fMX = testY.shape
    D = np.zeros((N + 1, M + 1))
    D[:, 0] = np.inf
    D[0, :] = 0
    S = np.zeros((N + 1, M + 1), dtype=np.int32)
    S[0, 1:M + 1] = np.arange(1, M + 1)
    S[:, 0] = 0
    opt_sub_seqs = np.zeros((M, 3))
    d_min = np.inf
    T_s = np.inf
    T_e = np.inf
    t = 0
    # ------Pad input data------------------------------------------------------
    queryX_new = np.zeros((N + 1, fNX))
    queryX_new[1:N + 1, :] = queryX
    queryX = queryX_new
    # queryX = np.concatenate((np.zeros((1,fNX)),queryX), axis=0)

    testY_new = np.zeros((M + 1, fMX))
    testY_new[1:M + 1, :] = testY
    testY = testY_new
    # testY = np.concatenate((np.zeros((1,fMX)),testY), axis=0)

    doyX_new = np.zeros((N + 1), dtype=np.int32)
    doyX_new[1:N + 1] = doyX
    doyX = doyX_new
    # doyX =  np.concatenate(([0],doyX), axis=0) #in MATLAB: doyX = [0;doyX];

    doyY_new = np.zeros((M + 1), dtype=np.int32)
    doyY_new[1:M + 1] = doyY
    doyY = doyY_new
    # doyY =  np.concatenate(([0],doyY), axis=0)#in MATLAB: doyY = [0;doyY];

    for j in range(1, M + 1):
        for i in range(1, N + 1):
            doy_xi = doyX[i]
            doy_yj = doyY[j]
            val_min = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            D[i, j] = (1.0 - theta) * li_nomr(queryX[i, :], testY[j, :]) + \
                      theta * computeTW(doy_xi, doy_yj, alpha, beta) + val_min
            place = get_index(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1], val_min)
            # place = [D[i-1,j], D[i,j-1], D[i-1,j-1]].index(val_min)
            ############## switch-case implementation
            if place == 0:
                S[i, j] = S[i - 1, j]
            elif place == 1:
                S[i, j] = S[i, j - 1]
            elif place == 2:
                S[i, j] = S[i - 1, j - 1]
        # The algorithm reports the subsequences after confirming that the
        # current optimal subsequences cannot be replaced by the upcoming
        # Subsequences. i.e. we report the subsequence that gives the minimum
        # distance d_min when the d(n/t,j) and s(n/t,j)arrarys saftify
        # For all j, (d(n,j) >= d_min OR s(n,j)>i_e/t_e)
        # --------Component I----------
        D_now = D[1:N + 1, j]
        S_now = S[1:N + 1, j]
        if d_min <= epsilon:
            flag = 0
            for ii in range(0, N):
                d_i = D_now[ii]
                s_i = S_now[ii]
                if d_i >= d_min or s_i > T_e:
                    flag = flag + 1
            if flag == N:
                # print("t=", t)
                opt_sub_seqs[t, :] = [d_min, T_s, T_e]
                # print(opt_sub_seqs)
                t = t + 1
                d_min = np.inf
                for ii in range(0, N):
                    if S_now[ii] <= T_e:
                        D_now[ii] = np.inf
        # --------Component II----------
        d_m = D_now[N - 1]
        # print("d_m = {} - epsilon = {} - d_min = {}".format(d_m,epsilon, d_min))
        if d_m <= epsilon and d_m < d_min:
            d_min = d_m
            T_s = S_now[N - 1]
            T_e = j - 1  # t = j-1
    D = D[1:N + 1, 1:M + 1]
    S = S[1:N + 1, 1:M + 1]
    opt_sub_seqs = opt_sub_seqs[0:t, :]
    check_value = opt_sub_seqs.shape[0]
    if (opt_sub_seqs.shape[0] != 0):
        dist_SPRING = np.min(opt_sub_seqs[:, 0])
    else:
        dist_SPRING = np.inf
    # Consider Tail effect
    dist_tail = np.inf
    if t > 0:
        T_e = np.max(opt_sub_seqs[:, 2]) + 2  # Here 3 refers to a neighsize, Neigh = 3;
        # print("T_e=", T_e)
        Tail_length = T_e - 1 + 0.3 * (N - 1)
        # print("T_e=", T_e, "-Tail_length=",Tail_length)
    else:
        T_e = 0
        Tail_length = 0
    if Tail_length <= M:
        Tail = np.array([np.inf] * M)
        Tail[int(T_e):len(Tail)] = D[N - 1, int(T_e):(D.shape[1])]
        # Tail_min = np.min(Tail)# min(list(Tail))
        t_e_Tail = np.argmin(Tail)  # min(list(Tail)) #list(Tail).index(min(list(Tail)))
        Tail_min = np.array([np.min(Tail), Tail[len(Tail) - 1]])
        dist_tail = np.min(Tail_min)
        T_place = np.argmin(Tail_min)
        if dist_tail <= epsilon:  # Tail includes a possible subsequence
            if T_place == 0:
                t_s_Tail = S[N - 1, t_e_Tail]
                dist_tail = D[N - 1, t_e_Tail]
            else:
                t_e_Tail = M - 1
                t_s_Tail = S[N - 1, M - 1]
                dist_tail = D[N - 1, M - 1]
            #################Apply Stack
            temp = np.zeros((opt_sub_seqs.shape[0] + 1, opt_sub_seqs.shape[1]))
            temp[0:opt_sub_seqs.shape[0], :] = opt_sub_seqs
            temp[opt_sub_seqs.shape[0]:opt_sub_seqs.shape[0] + 1, :] = np.array([1, 2, 3])
            opt_sub_seqs = temp
    I = np.argsort(opt_sub_seqs[:, 0])
    opt_sub_seqs = opt_sub_seqs[I, :]  # opt_sub_seqs[:,0].sort()
    #     # 3. Refine matches by removing redundant subsequences && Find path
    #     unique_sub_seqs = []
    #     path = []
    #     if (par_unique > 0) and (par_path):
    #         if "optimal" in par_path:
    #             opt_sub_seqs = opt_sub_seqs[0,:]
    #             unique_sub_seqs = opt_sub_seqs[:]
    #         else:
    #             u, ia = np.unique(opt_sub_seqs[:,1], return_index=True)
    #             unique_sub_seqs = opt_sub_seqs[ia,:]##### problem
    #         path = findPath_v2_jit(D,unique_sub_seqs,par_path)
    #     elif par_unique == 0 and (par_path):
    #         if "optimal" in par_path:
    #             opt_sub_seqs = opt_sub_seqs[0,:]
    #             unique_sub_seqs = opt_sub_seqs
    #         path = findPath_v2_jit(D,opt_sub_seqs,par_path)
    #     elif par_unique > 0 and not(par_path):
    #         u, ia = np.unique(opt_sub_seqs[:,1], return_index=True)
    #         unique_sub_seqs = opt_sub_seqs[ia,:]##### problem
    return (min(dist_SPRING, dist_tail, D[N - 1, M - 1]))

@jit(nogil=True)
def valid_param(X, Y):
    if (isinstance(X, list) and isinstance(Y, list)):
        return (1)
    elif (isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)):
        return (2)

@jit(nopython=True)
def computeTW(doy_x, doy_y, alpha, beta):
    ttl = abs(doy_x - doy_y)
    ttl1 = 1.0 + np.exp(-alpha * (ttl - beta))
    ttl2 = (1.0 / ttl1)
    return (ttl2)

@jit(nopython=True)
def li_nomr(X, Y):
    nn = X.shape[0]
    nnn = 0
    for z in range(0, nn):
        n = X[z] - Y[z]
        n2 = (n * n)
        nnn = nnn + (n2)
    return np.sqrt(nnn)

@jit(nopython=True)
def get_index(X, Y, Z, T):
    if X == T:
        return (0)
    elif Y == T:
        return (1)
    elif Z == T:
        return (2)


def twDTW_Spring_li_v2(queryX, doyX, testY, doyY, alpha, beta, theta, epsilon, par_unique, par_path):
    """regular twDTW_Spring without numba for optimization time execution
         Time weighted DTW with a SPRING version
                --------------------------------------------------------------------------
                 Input:
                       queryX: query sequence
                       testY: database/test sequence
                       doyX: datetime of X,
                       doyY: datetime of Y,
                       alpha, beta: parameters for weighted function [-0.1,100 by default]
                       theta: weight of time constraints, [0.5 by default]
                       epsilon: cost threshold [It equals the minimum distance when it is set to NULL]
                       par_path: par_path {[],"optimal","all",""}
                       par_unique: remove repeated, larger matches
                 Output:
                       D: accumlated cost matrix
                       S: position matrix
                       matches: [sorted] matches of subsequence
                       matches_refined: refined_matches
                       path: alignment of EITHER optimal subsequnce OR all subsequence
                 Reference:
                 [1] M?ller, Meinard. "Dynamic time warping." Information retrieval for music and motion (2007): 69-84.
                 [2] Sakurai, Yasushi, Christos Faloutsos, and Masashi Yamamuro. "Stream monitoring under the time warping distance." Data Engineering, 2007. ICDE 2007. IEEE 23rd International Conference on. IEEE, 2007.]
                 [2] trackback.m, inspired by Matlab
                 Copyright@ Mengmeng Li, ITC, University of Twente, 2018-02-20"""
    # --------------------------------------------------------------------------
    #                        Final version v2
    # --------------------------------------------------------------------------
    # 1. Parameter validation
    if (isinstance(queryX, np.ndarray) and isinstance(testY, list)):
        queryX = queryX[:]
        testY = testY[:]
    elif (isinstance(queryX, np.ndarray) and isinstance(testY, np.ndarray)):
        N, fNX = queryX.shape
        M, fMX = testY.shape
        if N < fNX or M < fMX:
            print("Make sure the signals are arranged column-wisely\n")
        if fNX != fMX:
            print("ERRORR !..Two inputs have different dimensions\n")
        if N > M:
            print("ERRORR !..Query sequence should not be longer than the test sequence\n")
    else:
        print("Only support vector or matrix inputs\n")
    #     # 2. SPRING DTW
    # 2.1 Define return variables
    N, fNX = queryX.shape
    M, fMX = testY.shape
    D = np.zeros((N + 1, M + 1))
    D[:, 0] = np.inf
    D[0, :] = 0
    S = np.zeros((N + 1, M + 1))
    S[0, 1:S.shape[1]] = range(1, M + 1)
    S[:, 0] = 0
    S.astype(int)
    opt_sub_seqs = np.zeros((M, 3))
    d_min = np.inf
    T_s = np.inf
    T_e = np.inf
    t = 0
    # ------Pad input data------------------------------------------------------
    queryX = np.concatenate((np.zeros((1, fNX)), queryX), axis=0)
    testY = np.concatenate((np.zeros((1, fMX)), testY), axis=0)
    doyX = np.concatenate(([0], doyX), axis=0)  # in MATLAB: doyX = [0;doyX];
    doyY = np.concatenate(([0], doyY), axis=0)  # in MATLAB: doyY = [0;doyY];
    # jit function
    # def dist_fun(queryX, testY, doyX, doyY, theta, alpha, beta, espilon, opt_sub_seqs, t, T_e, T_s, S, D, N, M):
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            doy_xi = doyX[i]
            doy_yj = doyY[j]
            oost = (1 - theta) * (np.linalg.norm(queryX[i, :] - testY[j, :])) + theta * computeTW(doy_xi, doy_yj, alpha,
                                                                                                  beta)
            bestMin = min([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            place = [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]].index(min([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]))
            D[i, j] = oost + bestMin
            ############## switch-case implementation
            key_S = {0: S[i - 1, j], 1: S[i, j - 1], 2: S[i - 1, j - 1]}
            S[i, j] = key_S.get(place, "default")

        # The algorithm reports the subsequences after confirming that the
        # current optimal subsequences cannot be replaced by the upcoming
        # Subsequences. i.e. we report the subsequence that gives the minimum
        # distance d_min when the d(n/t,j) and s(n/t,j)arrarys saftify
        # For all j, (d(n,j) >= d_min OR s(n,j)>i_e/t_e)

        # --------Component I----------
        D_now = D[1:N + 1, j]
        S_now = S[1:N + 1, j]
        epsilon
        if d_min <= epsilon:
            flag = 0  # Decide if all d_i satisfy the if-statement
            for ii in range(0, N):
                d_i = D_now[ii]
                s_i = S_now[ii]
                if d_i >= d_min or s_i > T_e:
                    flag = flag + 1
            if flag == N:
                # print("t=", t)
                opt_sub_seqs[t, :] = [d_min, T_s, T_e]
                # print(opt_sub_seqs)
                t = t + 1
                d_min = np.inf
                for ii in range(0, N):
                    if S_now[ii] <= T_e:
                        D_now[ii] = np.inf
                        # --------Component II----------
        d_m = D_now[N - 1]
        # print("d_m = {} - epsilon = {} - d_min = {}".format(d_m,epsilon, d_min))
        if d_m <= epsilon and d_m < d_min:
            # print("true")
            d_min = d_m
            T_s = S_now[N - 1]
            T_e = j - 1  # t = j-1

    D = D[1:N + 1, 1:M + 1]
    S = S[1:N + 1, 1:M + 1]
    opt_sub_seqs = opt_sub_seqs[0:t, :]

    if (opt_sub_seqs.any()):
        dist_SPRING = min(opt_sub_seqs[:, 0])
    else:
        dist_SPRING = np.inf;

    # Consider Tail effect
    dist_tail = np.inf
    if t > 0:
        T_e = max(opt_sub_seqs[:, 2]) + 2  # Here 3 refers to a neighsize, Neigh = 3;
        # print("T_e=", T_e)
        Tail_length = T_e - 1 + 0.3 * (N - 1)
        # print("T_e=", T_e, "-Tail_length=",Tail_length)
    else:
        T_e = 0
        Tail_length = 0

    if Tail_length <= M:
        Tail = np.array([np.inf] * M)
        Tail[int(T_e):len(Tail)] = D[N - 1, int(T_e):(D.shape[1])]
        # Tail_min = np.argpartition(Tail, k=2)[0]# min(list(Tail))
        t_e_Tail = np.argpartition(Tail, 2)[0]  # min(list(Tail)) #list(Tail).index(min(list(Tail)))
        Tail_min = Tail[t_e_Tail]
        dist_tail = min([Tail_min, Tail[len(Tail) - 1]])
        T_place = [Tail_min, Tail[len(Tail) - 1]].index(min([Tail_min, Tail[len(Tail) - 1]]))
        if dist_tail <= epsilon:  # Tail includes a possible subsequence
            # print("true")
            if T_place == 0:
                t_s_Tail = S[N - 1, t_e_Tail]
                dist_tail = D[N - 1, t_e_Tail]
            else:
                t_e_Tail = M - 1
                t_s_Tail = S[N - 1, M - 1]
                dist_tail = D[N - 1, M - 1]
            opt_sub_seqs = np.vstack((opt_sub_seqs, np.array([dist_tail, t_s_Tail, t_e_Tail])))
    # I = np.argsort(opt_sub_seqs, axis=0)
    # opt_sub_seqs = opt_sub_seqs[I[:,0],:]
    opt_sub_seqs = np.sort(opt_sub_seqs, axis=0)
    dist = min([dist_SPRING, dist_tail, D[N - 1, M - 1]])
    # 3. Refine matches by removing redundant subsequences && Find path
    unique_sub_seqs = []
    path = []
    if (par_unique > 0) and (par_path):
        if "optimal" in par_path:
            opt_sub_seqs = opt_sub_seqs[0, :]
            unique_sub_seqs = opt_sub_seqs[:]
        else:
            u, ia = np.unique(opt_sub_seqs[:, 1], return_index=True)
            unique_sub_seqs = opt_sub_seqs[ia, :]  ##### problem
        path = findPath_v2_jit(D, unique_sub_seqs, par_path)
    elif par_unique == 0 and (par_path):
        if "optimal" in par_path:
            opt_sub_seqs = opt_sub_seqs[0, :]
            unique_sub_seqs = opt_sub_seqs
        path = findPath_v2_jit(D, opt_sub_seqs, par_path)
    elif par_unique > 0 and not (par_path):
        u, ia = np.unique(opt_sub_seqs[:, 1], return_index=True)
        unique_sub_seqs = opt_sub_seqs[ia, :]  ##### problem
    return dist


def findPath_v2(dMat, matches, par_path):
    """This function finds the warping path of DTW using trace back strategy
    Function to find optimum path in
    Input:
          dMat: accumulated cost matrix
          matches: subsequence obtained by SPRING DTW
          par = "optimal": the optimal path
          par = "all": all the paths
    """
    path = dict()
    num_m = matches.shape[0]
    if num_m < 1:
        path = dict()
        return path
    if "optimal" in par_path:
        num_p = 1
    elif "all" in par_path:
        num_p = num_m
    else:
        print("Parameter setting for FIND PATH is not correct!")
    # pre-allocate to the maximum warping path size.
    [M, N] = dMat.shape
    ix = np.zeros((M + N))
    iy = np.zeros((M + N))
    for p in range(0, num_p):
        starting = matches[p, 1]  # starting point
        ending = matches[p, 2]  # ending point
        # end of path
        ix[0] = M
        iy[0] = ending
        i = M
        j = ending
        k = 1
        combined = np.zeros((M + N, 2))
        while i > 0 or j > starting:
            if ((j - starting) == 0):
                i = i - 1
            elif ((i - 1) == 0):
                j = j - 1
            else:
                min([dMat(i - 1, j), dMat(i, j - 1), dMat(i - 1, j - 1)])
                place = [dMat[i - 1, j], dMat[i, j - 1], dMat[i - 1, j - 1]].index(
                    min([dMat[i - 1, j], dMat[i, j - 1], dMat[i - 1, j - 1]]))
                key = {0: [i - 1, j], 1: [i, j - 1], 2: [i - 1, j - 1]}
                i, j = key.get(place, "default")
            k = k + 1
            ix[k] = i
            iy[k] = j
        ix = ix[(k - 1)::-1]
        iy = iy[(k - 1)::-1]
        combined[:, 0] = ix
        combined[:, 1] = iy
        path.update({p: combined})  ####path(p).path = [ix,iy];
        ix[:] = 0
        iy[:] = 0
    return path


findPath_v2_jit = numba.jit(findPath_v2)
