---
layout: post
title: "The Information Bottleneck Method"
description: review the information bottleneck method and the following-up papers to take a look IB framework's mainstream.
---


"The Information Bottleneck Method" is published in 1999 and had a lot of influence on deep learning. 
In this paper, the relevant information in a signal $$ x \in X $$ is defined by the information provides about another signal $$ y \in Y$$.
Understanding the signal $$ x $$ is not just predicting $$ y $$ through $$ x $$ but specifying which features of $$ X $$ take a role in.
Formally speaking, it is important to find a compact code for $$ X $$ that preserves the information about $$ Y $$ maximally.
In this regard, the variational principle in this paper provides a framework for dealing with various problems in signal processing.

Many of the sentences in this review were written by taking the contents of the original.

Besides some contents of the paper were skipped, because my knowledge was not sufficient to understand them and they were too complicated for my purpose of looking at the mainstream of the IB framework.


# Main

## 1. Relevant Quantization


- Notation

-- $$ X $$: the signal space with a fixed probability measure $$ p(x) $$

-- $$ \tilde{X} $$: quantized codebook of $$ X $$ (compressed representation of $$ X $$)


- Assumption

-- Both $$ X $$ and $$ \tilde{X} $$ are finite; i.e., any continuous space should be first quantized.

-- we have access to the $$ p(x) $$.


 $$ \forall x \in X $$, we seek a possibly stochastic mapping to a representative $$ \tilde{x} \in \tilde{X} $$ characterized by a conditional pdf $$ p(\tilde{x} \vert x) $$. 
This mapping $$ p(\tilde{x} \vert x) $$ induces a soft patrtitioning of $$ X $$ in which each block corresponds to each $$ \tilde{x} \in \tilde{X} $$, with probability given by

$$
\begin{aligned}
p(\tilde{x}) = \sum_x p(x)p(\tilde{x} \vert x). \qquad (1)
\end{aligned}
$$

The average volume of the elements of $$ X $$ that are mapped to the same codeword is $$ 2^{H(X \vert \tilde{X})} $$, and entropy $$ H(X) $$ is the expected amout of information to transmit information about random variable X.

What is this partitioning? For instance, in my opinion, it is similar with making groups among the objects for a specific purpose. 
Suppose that we need to classify 100 people in nationality. 
Then we will assign each of them to distinct sets.
These each person is $$ x \in X $$, distinct set is $$ \tilde{x} \in \tilde{X} $$ and nationality is, not mentioned yet, $$ y \in Y $$.

Let us consider the quality of a partitioning of $$ X $$.
The first factor of the quality of a quantization is the rate, or the average number of bits per message need to specify an element in the codebook without confusion.
The rate is bounded from below by the mutual information

$$
\begin{aligned}
I(X;\tilde{X}) &= H(X) - H(X \vert \tilde{X}) \\
	&= -\sum_x \sum_{\tilde{x}} p(x, \tilde{x}) log \frac{p(x)p(\tilde{x})}{p(x, \tilde{x})} \\
	&= -\sum_x \sum_{\tilde{x}} p(x, \tilde{x}) log \frac{p(\tilde{x} \vert x)}{p(x)}.
\end{aligned}
$$

However, only the rate is not enough score for a good quantization since $$ X $$ would shrink into one single point and the rate is fully minimized at that time.
Therefore we need at least one constraint.


### 1.1. Relevance through distortion: Rate distortion theory

In rate distortion theory, such a constraint is provided through a distortion function, $$ d:X \times \tilde{X} \rightarrow \mathbb{R}^+ $$ (smaller $$ d $$ is better $$ d $$).

The partitioning of $$ X $$ is induced by $$ p(\tilde{x} \vert x) $$ has an expected distortion

$$
\begin{aligned}
\langle d(x, \tilde{x}) \rangle_{p(x, \tilde{x})} = -\sum_x \sum_{\tilde{x}} p(x, \tilde{x})d(x, \tilde{x})
\end{aligned}
$$

The rate is in reverse proportion (monotonic tradeoff) to the expected distortion.

The rate distortion theorem of Shannon and Kolmogorov charaterizes this relationship through the rate distortion function, $$ R(D) $$, defined as the minimal achievable rate under a given constraint on the expected distortion:

$$
\begin{aligned}
R(D) \equiv \underset{ \{p(\tilde{x} \vert x) : \langle d(x, \tilde{x}) \rangle \leq D\} }{min}I(X;\tilde{X})
\end{aligned}
$$

Finding $$ R(D) $$ is a variational problem that can be solved by introducing a Lagrange multiplier, $$ \beta $$. One then need to minimize the functional

$$
\begin{aligned}
\mathcal{F}[p(\tilde{x} \vert x)] = I(X;\tilde{X}) + \beta\langle d(x, \tilde{x}) \rangle_{p(x, \tilde{x})}
\end{aligned}
$$

over all normalized(partitioned) distributions $$ p(\tilde{x} \vert x) $$. 
This formulation has the following consequences:


**Theorem 1.** *The solution of the variational problem,*

$$
\begin{aligned}
\frac{\delta \mathcal{F}}{\delta p(\tilde{x} \vert x)} = 0,
\end{aligned}
$$

*for normalized distributions $$ p(\tilde{x} \vert x) $$, is given by the exponential form*

$$
\begin{aligned}
p(\tilde{x} \vert x) = \frac{p(\tilde{x})}{Z(x, \beta)}exp[-\beta d(x,  \tilde{x})] \qquad (2)
\end{aligned}
$$

*where $$ Z(x, \beta) $$ is a normalization (partition) function. Moreover, the Lagrange multiplier $$ \beta $$, determined by the value of the expected distortion, $$ D $$ is positive and satisfies*

$$
\begin{aligned}
\frac{\delta R}{\delta D} = -\beta.
\end{aligned}
$$


### 1.2. The Blahut-Arimoto algorithm

Theorem 1 and Lemma 2 provides the Blahut-Arimoto algorithm, which is a converging iterative algorithm with a unique minimum for self-consistent determination of the distributions $$ p(\tilde{x} \vert x) $$ and $$ p(x) $$.

**Theorem 3.** *Equations (1) and (2) are satisfied simultaneously at the minimum of the functional,*

$$
\begin{aligned}
\mathcal{F} = -\langle \log Z(x, \beta) \rangle_{p(x)} = I(X;\tilde{X}) + \beta\langle d(x, \tilde{x}) \rangle_{p(x, \tilde{x})},
\end{aligned}
$$

*where the minimization is done independently over the convex sets of the normalized distributions $$ {p(\tilde{x})} $$ and $$ {p(\tilde{x} \vert x)} $$,*

$$
\begin{aligned}
\underset{p(\tilde{x})}{\min}\underset{p(\tilde{x} \vert x)}{\min}\mathcal{F}[p(\tilde{x});p(\tilde{x} \vert x)].
\end{aligned}
$$

*These independent conditions correspond precisely to alternating iterations of Eq. (1) and (2). Denoting by $$ t $$ the iteration step,*

$$
\begin{cases}
  p_{t+1}(\tilde{x}) = \sum_{x} p(x)p_{t}(\tilde{x} \vert x) \newline
  p_{t}(\tilde{x} \vert x) = \frac{p_{t}(\tilde{x})}{Z_{t}(x, \beta)}\exp(-\beta d(x, \tilde{x})) \qquad (3)
\end{cases}
$$

*where the normalization function $$ Z_{t}(x, \beta) $$ is evaluated for every t in Eq. (3). Furthermore, these itterations converge to a unique minimum of $$ \mathcal{F} $$ in the convex sets of the two distributions.*

It is important to note that **the BA algorithm does not search for an optimized representation $$ \tilde{X} $$,** but rather handle the optimal partitioning of $$ X $$ over a predefined representatives $$ \tilde{X} $$.


## 2. Relevance through another Variable: The Information Bottleneck

The problem of relevant quantization has to be addressed directly, by preserving the relevant information about another variable, since the right distortion measures is rarely available.

- Notation

-- $$ Y $$ : the another variable relevant to $$ X $$

- Assumption

-- $$ I(X;Y) > 0 $$.

-- we have access to the $$ p(x,y) $$.


### 2.1. A new variational principle

As chapter 1, we want to minimize the rate as much as possible.
On the other hand, we now want this quantization to capture as much of the information about $$ Y $$; 
i.e., the constraint is changed from minimizing $$ \langle d(x, \tilde{x}) \rangle_{p(x, \tilde{x})} $$ to maximizing $$ I(\tilde{X};Y) $$.

As with rate and distortion, there is a tradeoff between compressing the representation and preserving meaningful information, and there is no single right solution for the tradeoff.
The assignment we are looking for is the one that keeps a fixed amount of meaningful information about the relevant signal $$ Y $$ while minimizing the number of bits from the original signal $$ X $$.
In effect, **we pass the information that $$ X $$ provides about $$ Y $$ through a "bottleneck" formed by the compact summaries in $$ \tilde{X} $$.**

We can find the optimal assignment by minimizing the functional

$$
\begin{aligned}
\mathcal{L}[p(\tilde{x} \vert x)] = I(X;\tilde{X}) - \beta I(\tilde{X};Y),
\end{aligned}
$$

At $$ \beta=0 $$, $$ X $$ shrinked into a single point, while as $$ \beta \rightarrow \infty $$ we got arbitrarily trivial quantization with full of redundancy,
Interestingly, if there are **sufficient statistics**, it is possible to preserve almost all the meaningful information at finite $$ \beta $$ with a significant compression of the original data.


### 2.2. Self-consistent equations & The information bottleneck iterative algorithm 

Like the rate distortion problem, for this new variational principle, there are theorem 4 and the information bottleneck iterative algorithm, corresponding to theorem 1 and the BA algorithm, respectively.

In contrast to BA algorithm, the IB algorithm does not imply uniqueness of the solution.


### 2.3. The structure of the solutions

The formal solution of the self consistent equations **still** requires a specification of the structure and cardinality of $$ \tilde{X} $$, as in rate distortion theory.
For every value of the Lagrange multiplier $$ \beta $$, and for every choice of the cardinality of $$ \tilde{X} $$, there are corresponding values of the mutual information $$ I_X \equiv I(X, \tilde{X}) $$ and $$ I_Y \equiv I(\tilde{X}, Y) $$.
The variational principle implies that 

$$
\begin{aligned}
\frac{I(\tilde{X}, Y)}{I(X, \tilde{X})} = \beta^{-1} > 0,
\end{aligned}
$$

which suggests a *deterministic annealing* approach.
By increasing the value of $$ \beta $$ one can move along *convex* curves in the "information plane" $$ (I_X, I_Y) $$.
These curves, analogous to the rate distortion curves, exists for every choice of the cardinality of $$ \tilde{X} $$.
The solutions of the self consistent equations thus correspond to a family of such annealing curves, all starting from the (trivial) point (0, 0) in the information plane with infinite slope and parameterized by $$ \beta $$.



## Reference

- Tishby, Naftali, Fernando C. Pereira, and William Bialek. "The information bottleneck method." arXiv preprint physics/0004057 (2000).