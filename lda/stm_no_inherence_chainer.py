#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# (c)2018 Hiroki Iida / Retrieva Inc.

import numpy as np
import pandas as pd
import sklearn
import glmnet_python
from collections import Counter
import lda

class STM:
    def __init__(self, K, X, Y, docs, V, sigma, smartinit=True, interact=True):
        self.A = np.unique(np.array(Y))
        self.X = X # DxPx matrix (Px is the num of tags)
        self.Y = Y # Dx1 matrix
        self.K = K
        self.D = len(docs)

        if X:
            P = X.shape[1]
            self.Gamma = np.zeros([P, K-1]) # parameter of topics prior

        self.mu = np.zeros([self.D, K])
        self.Sigma = np.diag(np.ones(K-1)) * sigma # if zero, no update. so using diag.
        self.c_dv = np.zeros([self.D, V], dtype=int)
        self.W = np.zeros(self.D, dtype=int)
        mv = np.zeros(V, dtype=int)
        for m, doc in enumerate(docs):
            for t in doc:
                self.c_dv[m, t] += 1
                self.W[m] += 1
                mv[t] += 1

        self.mv = np.log(mv)

        self.docs = docs

        self.V = V
        self.eta = np.zeros([self.D, K])
        self.theta = self.eta2theta(self.eta)
        self.q_z = np.zeros([self.D, self.K, self.V])

        if self.Y:
            if interact:
                self.coef_row = self.A * self.K + self.A + self.K
            else:
                self.coef_row = self.A * self.K

            self.kappa_all = np.zeros([self.coef_row, V])
            self.phi = np.ones([self.A, self.K, self.V]) / self.V

            ## for update phi
            K_diag = np.diag(np.ones(self.K))
            K_A_diag = K_diag
            for _ in range(self.A):
                K_A_diag = np.concatenate(K_A_diag, K_diag, axis=0)

            A_col = np.zeros(self.K * self.A, self.A)
            att = np.ones(self.K)
            for i, a in enumerate(range(self.A)):
                start = i * self.K
                end = (i + 1) * self.K
                A_col[start:end, i * K, a] = input

            self.cov = np.concatenate(K_A_diag, A_col, axis=1)
            if interact:
                np.diag(np.ones(K * self.A))
                self.cov = np.concatenate(self.cov, A_col)

        else:
            self.phi = np.ones([self.K, self.V]) / self.V

    def eta2theta(self, eta):
        return np.exp(eta) / np.sum(np.exp(eta), axis=1)[:, np.newaxis]

    def lda_initialize(self, alpha, beta, itr, voca, smartinit=True):
        lda_init = lda.LDA(self.K, alpha, beta, self.docs, self.V, smartinit)
        lda.lda_learning(lda_init, itr, voca)

        Kalpha = self.K * alpha
        for m, doc in enumerate(self.docs):
            self.theta[m] = lda_init.n_m_z[m] / (len(self.docs[m]) + Kalpha)
        if self.Y:
            for a in range(self.A):
                self.phi[a] = lda_init.worddist()
            for m, a in enumerate(self.Y):
                self.q_z[m] = self.theta[m][:, np.newaxis] * self.phi[a]
                self.q_z[m] /= np.sum(self.q_z[m], axis=0)
        else:
            self.phi = lda_init.worddist()
            for m, doc in enumerate(self.docs):
                self.q_z[m] = self.theta[m][:, np.newaxis] * self.phi
                self.q_z[m] /= np.sum(self.q_z[m], axis=0)

    def inference(self):
        """learning once iteration"""
        # E-step
        ## update q_eta and q_z
        print("theta_sum", np.sum(self.theta, axis=1))
        print("theta", self.theta, np.any(np.isnan(self.theta)), np.any(self.theta >= 0.0))
        print("phi_sum", np.sum(self.phi, axis=1))
        print("phi", self.phi, np.any(np.isnan(self.phi)), np.any(self.phi >= 0.0))
        print("q_z", np.sum(self.q_z, axis=1), self.q_z, np.any(np.isnan(self.q_z)))
        print("mu", self.mu)
        print("Sigma",self.Sigma)
        if self.Y:
            for m, (_, a) in enumerate(zip(self.docs, self.Y)):
                eta_tmp = np.sum(self.c_dv[m] * self.q_z[m], axis=1) - self.W[m] * self.theta[m]
                self.eta[m] = self.mu[m] + np.append(np.dot(self.Sigma, eta_tmp[:self.K-1]), 0.0)
                self.theta[m] = np.exp(self.eta[m]) / np.sum(np.exp(self.eta[m])) # K-vec / scaler
                self.q_z[m] = np.exp(self.eta[m])[:, np.newaxis] * self.phi[a]
                self.q_z[m] /= np.sum(self.q_z[m], axis=0) # summing kxv matrix
        else:
            for m, _ in enumerate(self.docs):
                eta_tmp = np.sum(self.c_dv[m] * self.q_z[m], axis=1) - self.W[m] * self.theta[m]
                self.eta[m] = self.mu[m] + np.append(np.dot(self.Sigma , eta_tmp[:self.K-1]), 0.0)
                print("eta_tmp:",eta_tmp)
                print("eta[{}]:{}".format(m, self.eta[m]))
                self.theta[m] = np.exp(self.eta[m]) / np.sum(np.exp(self.eta[m]))
                self.q_z[m] = np.exp(self.eta[m])[:, np.newaxis] * self.phi
                self.q_z[m] /= np.sum(self.q_z[m], axis=0)

        # M-step
        ## update Gamma
        if self.X:
            self.Gamma = RVM_regression(self.eta, self.Gamma, self.X, self.K)
            self.mu = np.dot(self.X, self.Gamma)
        else:
            mu_prev = np.zeros([self.D, self.K]) + self.mu
            self.mu = np.tile(np.average(self.eta, axis=0), (self.D,1))

        ## update Sigma
        ### prepare update Sigma(calc q_v) and phi(calc phi_tmp)
        if self.Y:
            E_count = np.zeros(self.A, self.K, self.V)
            q_v = np.zeros([self.D, self.K - 1, self.K - 1])
            tmp_for_sigma = np.zeros([self.D, self.K - 1, self.K - 1])
            for m, (_, a) in enumerate(zip(self.docs, self.Y)):
                tmp = np.zeros([self.V, self.K-1, self.K-1])
                E_count[a] = self.c_dv[m] * self.phi[a]
                for v in range(self.V):
                    tmp[v] += np.outer(self.phi[a, 0:self.K-1, v], self.phi[a, 0:self.K-1, v])

                hessian = (np.diag(np.dot(self.c_dv[m], self.phi[a, 0:self.K-1, :])) - np.sum(self.c_dv[m][:, np.newaxis, np.newaxis] * tmp, axis=0)) - \
                        self.W[m] * (np.diag(self.theta[m, 1:self.K-1]) - np.outer(self.theta[m, 0:self.K-1], self.theta[m, 0:self.K-1])) - np.linalg.inv(self.Sigma)
                q_v[m] -= np.linalg.inv(hessian)

                if self.X:
                   tmp_for_sigma_outer = self.eta[m, 0:self.K-1] - np.dot(self.X, self.Gamma)[m]
                   tmp_for_sigma[m] += np.outer(tmp_for_sigma_outer, tmp_for_sigma_outer)
                else:
                    tmp_for_sigma_outer = self.eta[m, 0:self.K-1] - self.mu
                    tmp_for_sigma[m] += np.outer(tmp_for_sigma_outer, tmp_for_sigma_outer)

        else:
            q_v = np.zeros([self.D, self.K - 1, self.K - 1])
            tmp_for_sigma = np.zeros([self.D, self.K - 1, self.K - 1])
            for m, _ in enumerate(self.docs):
                hessian = np.diag(-1.0 * np.dot(self.q_z[m, 0:self.K-1, :], self.c_dv[m])) \
                          + np.dot(np.sqrt(self.c_dv[m]) * self.q_z[m, 0:self.K-1, :], (np.sqrt(self.c_dv[m]) * self.q_z[m, 0:self.K-1, :]).T) \
                          + self.W[m] * np.diag(self.theta[m, 0:self.K-1]) \
                          - self.W[m] * np.outer(self.theta[m, 0:self.K-1], self.theta[m, 0:self.K-1]) + np.linalg.inv(self.Sigma)

                q_v[m] = np.linalg.inv(hessian)
                print("q_v[{}]:{}".format(m, np.diag(q_v[m]) >= 0.0))

                if self.X:
                    tmp_for_sigma_outer = self.eta[m, 0:self.K-1] - np.dot(self.X, self.Gamma)[m]
                    tmp_for_sigma[m] += np.outer(tmp_for_sigma_outer, tmp_for_sigma_outer)
                else:
                    tmp_for_sigma_outer = self.eta[m, 0:self.K-1] - self.mu[m, 0:self.K-1]
                    tmp_for_sigma[m] += np.outer(tmp_for_sigma_outer, tmp_for_sigma_outer)
                    #print("tmp_sigma{}:{}".format(m, tmp_for_sigma[m]))

            # self.Sigma = np.average(tmp_for_sigma, axis=0)
            self.Sigma = np.average(q_v + tmp_for_sigma, axis=0)

        ## update phi
        if self.Y:
            coef = np.zeros(self.coef_row, self.V)
            for v in range(self.V):
                mod = glmnet_python.glmnet(x=self.cov, y=E_count[:,:,v], family='poisson', alpha=1.0,
                                           offset=self.mv[v], standardize=False, lambda_min=0.001,
                                           nlambda=250,  maxit=100, thresh=1e-5)
                dev = (1 - mod['dev'])*mod['nulldev']
                ic = dev + 2.0 * mod['df']
                ic_min_id = np.argmin(ic)
                coef[:,v] += mod['beta'][ic_min_id]

            self.kappa_all = coef
            phi_tmp = np.exp(np.dot(self.cov, coef) + self.mv)
            phi_tmp /= np.sum(phi_tmp, axis=1)
            for a, v in enumerate(phi_tmp.split(phi_tmp, self.K, axis=0)):
                self.phi[a] = v
        else:
            # ref: Variational EM Algorithms for Correlated Topic Models / Mohhammad Emtiaz Khan et al
            for k in range(self.K):
                self.phi[k] = np.sum(self.c_dv * self.q_z[:,k,:], axis=0)

            self.phi /= np.sum(self.phi, axis=1)[:, np.newaxis]

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None, Y=None):
        if docs == None: docs = self.docs
        if Y == None: Y = self.Y
        log_per = 0
        N = 0
        if Y:
           for m, (doc , a) in enumerate(zip(docs, Y)):
               for w in doc:
                   log_per -= np.log(np.dot(self.phi[a,:,w], self.theta[m]))
               N += len(doc)
        else:
           for m, doc in enumerate(docs):
               for w in doc:
                   log_per -= np.log(np.dot(self.phi[:,w], self.theta[m]))
               N += len(doc)
        return np.exp(log_per / N)

def RVM_regression(Y, W, X, K, it_num=100):
    """
    Parameters
    ---------
    Y: NxK matrix of target value

    W: DxK matrix of weight of linear regression

    X: NxD matrix of data

    K: topic number

    it_num: repeat count

    sup: N is data number(so it is equivalent to document number)
         D is data-dimension

    Returns:
    --------
    W: updated weight of linear regression
subM <- function(x, p) {
    ind <- (x@p[p]+1):x@p[p+1]
    rn <- x@i[ind]+1
    y <- x@x[ind]
    out <- rep(0, length=nrow(x))
    out[rn] <- y
    out
}

    ref: VB INFERENCE FOR LINEAR/LOGISTIC REGRESSION JAN DRUGOWITSCH et al
    """
    # inv-gamma prior from thesis
    N = X.shape[0]
    D = X.shape[1]

    a0 = np.ones(K) * 0.01
    b0 = np.ones(K) * 0.0001
    c0 = np.ones(K) * 0.01
    d0 = np.ones(K, D) * 0.001

    a_N = a0
    b_N = b0
    c_N = c0
    d_N = d0
    const_inv_V_N = np.dot(X.T, X)
    const_W = np.zeros(K)
    for k in range(K):
        const_W[k] = np.dot(X.T, Y[:,k])

    const_Y = np.sum(Y*Y, axis=0)

    for _ in range(it_num):
        inv_V_N = np.zeros([K, D, D])
        for k in range(K):
            inv_V_N[k, :, :] += np.diag(np.ones(D) * c_N[k] / d_N[k, :]) + const_inv_V_N

        for k in range(K):
            W[:, k] = np.dot(np.linalg.inv(inv_V_N[k]), np.dot(X.T, Y[:, k]))

        a_N = a0 + 0.5 * N
        for k in range(K):
            b_N[k] = b0 + 0.5 * (const_Y - np.dot(W[:, k], np.dot(inv_V_N[k], W[:, k].T)))

        c_N = c0 + 0.5
        for k in range(K):
            d_N[k] = d0 + 0.5 * W[:, k] * W[:, k] * a_N[k] / b_N[k]

        """
        L_Q =
        print("RVM_regression LVLO={}'.format(L_Q)
        """

    return W

def stm_learning(stm, iteration, voca):
    pre_perp = stm.perplexity()
    print ("initial perplexity=%f" % pre_perp)
    for i in range(iteration):
        stm.inference()
        perp = stm.perplexity()
        print ("-%d p=%f" % (i + 1, perp))
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(stm, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(stm, voca)

def output_word_topic_dist(stm, voca):
    if stm.Y:
        phi = np.sum(stm.phi, axis=0) * 0.5
        for k in range(stm.K):
            print("\n-- topic: {}".format(k))
            for w in np.argsort(-phi[k])[:20]:
                print("{}: {}".format(voca[w], phi[k,w]))
    else:
        phi = stm.phi
        for k in range(stm.K):
            print("\n-- topic: {}".format(k))
            for w in np.argsort(-phi[k])[:20]:
                print("{}: {}".format(voca[w], phi[k,w]))

def main():
    import argparse
    import vocabulary
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filename", help="corpus filename csv")
    parser.add_argument("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_argument("--alpha", dest="alpha", type=float, help="parameter alpha", default=1.0)
    parser.add_argument("--beta", dest="beta", type=float, help="parameter beta", default=0.1)
    parser.add_argument("-k", dest="K", type=int, help="number of topics", default=20)
    parser.add_argument("-i", dest="iteration", type=int, help="iteration count", default=100)
    parser.add_argument("-x", dest="X", type=str, help="preverence column", default=None)
    parser.add_argument("-y", dest="Y", type=str, help="covariate column", default=None)
    parser.add_argument("--sigma", dest="sigma", help="value of sigma()", default=0.01)
    parser.add_argument("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_argument("--seed", dest="seed", type=int, help="random seed")
    parser.add_argument("--df", dest="df", type=int, help="threshold of document freaquency to cut words", default=0)
    parser.add_argument("--interact", dest="interact", help="consider covariate interaction", default=False)
    parser.add_argument("--sinit", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
    parser.add_argument("--jap", dest="japanese", action='store_true', help='for japanese', default=False)
    options = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        load_doc = pd.read_csv(options.filename)
        if options.japanese:
            corpus = vocabulary.load_corpus_ja(load_doc['documents'])
        else:
            corpus = vocabulary.load_dataframe(load_doc['documents'])
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")

    if options.seed != None:
        np.random.seed(options.seed)

    print("proc voca")
    voca = vocabulary.Vocabulary(options.stopwords)
    print(corpus)
    docs = [voca.doc_to_ids(doc) for doc in corpus]

    ## process prevarence, if it is pointed
    print("proc X")
    if options.X:
        X = pd.Categorical(load_doc[options.X.split(',')])
    else:
        X = options.X

    ## process covariate, if it is pointed
    print("proc Y")
    if options.Y:
        Y = pd.Categorical(load_doc[options.Y])
    else:
        Y = options.Y

    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    print("set STM obj")
    stm = STM(options.K, X, Y, docs, voca.size(), options.sigma, options.smartinit)
    print ("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta))

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    print("lda_initialize")
    stm.lda_initialize(options.alpha, options.beta, 30, voca, options.smartinit)
    print("stm_learning")
    stm_learning(stm, options.iteration, voca)

if __name__ == "__main__":
    main()
