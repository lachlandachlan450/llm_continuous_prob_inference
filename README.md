# Abstract
Probabilistic inference in language models requires estimating expec-
tations over discrete token distributions, a computationally intensive
task given the large vocabularies involved. We propose a novel approach
that exploits the differentiability of many properties of interest with
respect to the token embedding space to achieve more efficient proba-
bilistic inference. For such properties, we construct efficient Sequential
Monte Carlo (SMC) estimators that operate in the continuous domain
rather than among discrete tokens, thereby avoiding some difficulties
associated with performing MCMC on large discrete distributions, and
provide some theoretical convergence guarantees for these estimators.
We apply our methodology to the problem of Low Probability Esti-
mation (LPE), introduced by Wu and Hilton (2024), which aims to
quantify the probability of rare outputs from language models. Our
approach enables efficient precise measurement of infrequent but po-
tentially significant generations, such as those that might represent
harmful or unexpected model behaviors.
