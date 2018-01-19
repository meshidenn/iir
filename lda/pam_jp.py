#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Pachinko Allocation Model + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# (c)2018 Hiroki Iida / Retrieva Inc.

import numpy as np


class PAM:
    def __init__(self, S, K, alphas, alphak, beta, docs, V, smartinit=True):
        self.S = S  # super topic num
        self.K = K  # sub topic num
        self.alphas = np.ones(S) * alphas  # parameter of super-topic prior
        self.alphask = np.ones(S, K) * alphak  # parameter of sub-topic prior
        self.beta = np.ones(K) * beta   # parameter of words prior
        self.docs = docs  # matrix of each word id for each docs
        self.V = V  # Number of all vocaburary
        # topics of words of documents
        # word count of each document and topic
        # zs is super-topic, zk is sub-topic
        # m is docnum, and j is position of vocaburary in doc
        # if count is reguralized, it turns into prob distribution
        self.zs_m_j = []
        self.zk_m_j = []
        self.n_m_zs = np.zeros((len(self.docs), S))
        self.n_m_zk = np.zeros((len(self.docs), S, K))
        # count of each topic and vocabulary
        self.n_zk_t = np.zeros((K, V))
        # topic count of each topic
        self.n_zk = np.zeros(K)
        self.N = len(doc)
        for m, doc in enumerate(docs):
            self.N += len(doc)  # all word
            zs_n = []  # super-topic of nth word in mth doc
            zk_n = []  # sub-topic of nth word in mth doc
            for t in doc:  # t is id of word
                if smartinit:
                    n_s = self.n_m_zs[m] + self.alphas
                    p_s = n_s / np.sum(n_s)
                    n_k = self.n_m_zk[m] + self.alphask
                    p_k = n_k / np.sum(n_k, axis=1)
                    n_v = self.n_zk_t[t] + self.beta
                    p_v = n_v / (self.n_zk + self.beta)
                    p_zsk = p_s * p_k * p_v

                    p_zs = np.sum(p_zsk, axis=1)
                    p_zk = np.sum(p_zsk, axis=0)

                    zs = np.random.multinomial(1, p_zs).argmax()
                    zk = np.random.multinomial(1, p_zk).argmax()

                else:
                    zs = np.random.randint(0, S)
                    zk = np.random.randint(0, K)

                self.n_m_zs[m, zs] += 1  # set super topic count on each doc
                zs_n.append(zs)
                self.n_m_zk[m, zk] += 1  # set sub topic count on each doc
                zk_n.append(zk)  # set topic on each word
                self.n_zk_t[zk, t] += 1  # set wordcount on each topic
                self.n_zk[zk] += 1  # set topic distribution
            self.zs_m_n.append(np.array(zs_n))
            self.zk_m_m.append(np.array(zk_n))

    def inference(self):
        """learning once iteration using collapsed Gibbs Sampling"""
        for m, doc in enumerate(self.docs):
            # Be careful followings are views
            # So self.hoge will be change, when changing variant
            zs_j = self.zs_m_n[m]
            zk_j = self.zk_m_n[m]
            n_m_zs = self.n_m_zs[m]
            n_m_zk = self.n_m_zk[m]
            for j, t in enumerate(doc):
                # discount for n-th word t with topic z
                zs = zs_j[j]
                zk = zk_j[j]
                n_m_zs[zs] -= 1
                n_m_zk[zk] -= 1
                self.n_zk_t[zk, t] -= 1
                self.n_z[zk] -= 1

                # sampling topic new_z for t
                n_s = n_m_zs + self.alphas
                p_s = n_s / np.sum(n_s)
                n_k = n_m_zk + self.alphask
                p_k = n_k / np.sum(n_k, axis=1)
                n_v = self.n_zk_t[t] + self.beta
                p_v = n_v / (self.n_z[zk] + self.beta)
                p_zsk = p_s * p_k * p_v  # SxK matrix

                p_zs = np.sum(p_zsk, axis=1)
                p_zk = np.sum(p_zsk, axis=0)

                new_zs = np.random.multinomial(1, p_zs).argmax()
                new_zk = np.random.multinomial(1, p_zk).argmax()


                # set z the new topic and increment counters
                zs_j[j] = new_zs
                zk_j[j] = new_zk
                n_m_zs[new_zs] += 1
                n_m_zk[new_zk] += 1
                self.n_m_zs
                self.n_zk_t[new_zk, t] += 1
                self.n_zk[new_zk] += 1

    def hyper_parameter_inference(self):
        mean_sk = np.average(self.n_m_zk / self.n_m_zs, axis=0)
        var_sk = np.var(self.n_m_zk / self.n_m_zs, axis=0)
        m_sk = mean_sk * (mean_sk - 1) / var_sk - 1
        self.alphas = np.exp(np.sum(np.log(m_sk), axis=1) / (self.K - 1)) / 5
        self.alphask = mean_sk / self.alphas

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, np.newaxis]

    def perplexity(self, docs=None):
        if docs is None:
            docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:, w], theta))
            N += len(doc)
        return np.exp(log_per / N)


def pam_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    print("initial perplexity=%f" % pre_perp)
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print("-%d p=%f" % (i + 1, perp))
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca)


def output_word_topic_dist(lda, voca):
    zcount = np.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in range(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    for k in range(lda.K):
        print("\n-- topic: %d (%d words)" % (k, zcount[k]))
        for w in np.argsort(-phi[k])[:20]:
            print("%s: %f (%d)" % (voca[w], phi[k, w], wordcount[k].get(w, 0)))


def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus",
                      help="using range of Brown \
                      corpus' files(start:end)")
    parser.add_option("-d", dest="dirname", help="corpus directory")
    parser.add_option("--alphas", dest="alphas",
                      type="float", help="parameter \
                      alphas", default=0.5)
    parser.add_option("--alphak", dest="alphak",
                      type="float", help="parameter \
                      alphak", default=0.5)
    parser.add_option("--beta", dest="beta",
                      type="float", help="parameter \
                      beta", default=0.5)
    parser.add_option("-s", dest="S", type="int",
                      help="number of super topics",
                      default=5)
    parser.add_option("-k", dest="K", type="int",
                      help="number of sub topics",
                      default=20)
    parser.add_option("-i", dest="iteration",
                      type="int", help="iteration \
                      count", default=100)
    parser.add_option("-si", dest="smartinit",
                      action="store_true", help="smart\
                      initialize of parameters",
                      default=False)
    parser.add_option("-j", dest='japanese',
                      action="store_true",
                      help="input japanese document")
    parser.add_option("--stopwords", dest="stopwords",
                      help="exclude stop words",
                      action="store_true",
                      default=False)
    parser.add_option("--seed", dest="seed",
                      type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int",
                      help="threshold of document \
                      freaquency to cut words", default=0)
    parser.add_option("-e", dest="e", action='store_true',
                      help="hyper parameter inference")
    (options, args) = parser.parse_args()
    if not (options.dirname or options.filename or options.corpus):
        parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.dirname and options.japanese:
            corpus = vocabulary.load_file_ja(options.dirname)
    elif options.filename:
            corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus:
            parser.error("corpus range(-c) forms 'start:end'")
    if options.seed is not None:
        np.random.seed(options.seed)

    voca = vocabulary.Vocabulary(options.stopwords)
    print(corpus)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0:
        docs = voca.cut_low_freq(docs, options.df)

    pam = PAM(options.S, options.K, options.alphas, options.alphak,
              options.beta, docs, voca.size(), options.smartinit)
    print("corpus=%d, words=%d, K=%d, a=%f, b=%f"
          % (len(corpus), len(voca.vocas), options.K,
             options.alpha, options.beta))

# import cProfile
# cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(),locals(), 'lda.profile')
    pam_learning(pam, options.iteration, voca)


if __name__ == "__main__":
    main()
