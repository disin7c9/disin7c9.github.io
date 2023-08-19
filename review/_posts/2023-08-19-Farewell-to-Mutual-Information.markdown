---
layout: post
title: "Farewell to Mutual Information: Variational Distillation for Cross-Modal Person Re-Identification"
description: review the information bottleneck method and the following-up papers to take a look IB framework's mainstream.
---


The Information Bottleneck framework proposes a information theoretical principle in the representation learning which preserves information of input data relevant to label while removes irrelevant one.
Despite of the wide range of applications, the optimization of the IB needs the precise estimation of mutual information.
In the paper, the authors present a novel strategy, Variational Self-Distillation (VSD), which does not explicitly estimate MI as well as provide analystic and scalable solution.
Furthermore, by expending VSD to mutli-view learning, they provide Variational Mutual-Learning (VML) and Variational Cross-Distillation (VCD) which remove view-specific and task-irrelevant information respectively.


# Main

One of the shortcomings of the conventional IB methods is that they have to compromise between representation compression and Empirical Risk Minimization (ERM), which makes it difficult to achieve both compression and prediction of flawless quality.
The authors present a new approach that makes it possible to preserve sufficient amout of relevant information meanwhile remove irrelevant redundancy.
The most important property of this approach is that **the approach optimizes IB based on MI, but does not explicitly estimates it.**


- Notation

-- $$ x $$: input data

-- $$ y $$: the label of $$ x $$

-- $$ v $$: an observation of $$ x $$ extracted from an encoder $$E(v \vert x)$$ which contains all infomation of y

-- $$ z $$: an extra code of $$ x $$ extracted from another encoder $$E'(z \vert v)$$ (which preserves all relevant information and simultaneously discards the irrelevant.)


## 1. Variational Self-Distillation

**According to the paper, many of the concepts and proofs from here rely on Ref. 2.**


**Definition 1. Sufficiency:** *$$ z $$ is sufficient for $$ y $$ if and only if*

$$
\begin{aligned}
I(z;y) = I(v;y).
\end{aligned}
$$

Let us consider the mutual information (MI) between $$ v $$ and $$ z $$.

$$
\begin{aligned}
I(v;z) = I(z;y) + I(v;z \vert y),
\end{aligned}
$$

where $$ I(z;y) $$ is relevant information and $$ I(v;z \vert y) $$ is irrelevant.
To make $$ z $$ an approximation of **minimal sufficient statistics** for $$ y $$, we need to maximize $$ I(z;y) $$ and minimize $$ I(v;z \vert y) $$.
Due to **Data Processing Inequality** $$ I(z;y) \leq I(v;y) $$, we have:

$$
\begin{aligned}
I(v;z) \leq I(v;y) + I(v;z \vert y).
\end{aligned}
$$

Therefore it is necessary for preserving sufficiency to maximize $$ I(v;y) $$ first and then minimize $$ I(v;y) - I(z;y) $$.

In this view, optimizing sufficiency of $$ z $$ for $$ y $$ is reformulated as 3 sub-optimizations:

1st. $$ \max I(v;y) $$

2nd. $$ \min \{ I(v;y) - I(z;y) \}$$

3rd. $$ \min I(v;z \vert y)$$.

Since the 1st is not related to $$ z $$ AND the 2nd and the 3rd are equivalent, these sub-optimizations is simplified to:

$$ 
\begin{aligned}
\min \{ I(v;y) - I(z;y) \}.
\end{aligned}
$$

However, it is hard to estimate MI in high dimension.

From the definition of MI, we can change the optimization problem into the following form.

$$ 
\begin{aligned}
\min \{ H(y \vert z) - H(y \vert v) \}. 
\end{aligned}
$$


**Corollary 1.** *If the Kullback-Leibler divergence between the predicted distributions of a sufficient observation $$v$$ and the representation $$z$$ equals to 0, then $$z$$ is sufficient for $$y$$ as well; i.e.,*

$$ 
\begin{aligned}
KL(p(y \vert v) \parallel p(y \vert z)) = 0 \quad \Rightarrow \quad H(y \vert z) - H(y \vert v) = 0.
\end{aligned}
$$

Consequently, sufficiency of $$z$$ for $$y$$ could be achieved by the following loss function:

$$ 
\begin{aligned}
\mathcal{L}_{VSD} = \underset{\theta, \phi}{\min} \mathbb{E}_{v \sim E_{\theta}(v \vert x)} [\mathbb{E}_{z \sim E_{\phi}(z \vert v)}[KL(p(y \vert v) \parallel p(y \vert z))]],
\end{aligned}
$$

where $$ \theta $$ and $$ \phi $$ are parameters of encoder and IB respectively.
By observing the optimization, this approach is a self-distillation method that remove irrelevant information.


## 2. Variational Cross-Distillation and Variational Mutual-Learning


Multi-view representation learning has been gaining attention as more and more real-world data are obtained from various sources and feature extractors.

- Notation

-- $$v_1$$&$$v_2$$: observations of $$x$$ from different view points

- Assumption

-- $$v_1$$&$$v_2$$ are sufficient for y.


### 2.1. VML

From the assumption, for any $$z$$ contains relevant information from the intersection of $$v_1$$ and $$v_2$$, it would eliminates view-specific information.
Motivated by this, define **consistency** w.r.t. $$z_1,z_2$$.

**Definition 2. Consistency:** *$$z_1$$ and $$z_2$$ are view-consistent if and only if*

$$ 
\begin{aligned}
I(z_1;y) = I(v_1v_2;y) = I(z_2;y)
\end{aligned}
$$

Intuitively, view-consistency between $$z_1$$ and $$z_2$$ means that they have the same amount of information about the labels.

Let us consider the MI between $$v_1$$ and $$z_1$$.

$$ 
\begin{aligned}
I(v_1;z_1) = I(v_1;z_1 \vert v_2) + I(z_1;v_2),
\end{aligned}
$$

where $$I(v_1;z_1 \vert v_2)$$ is view-specific information passed from $$v_1$$ exclusive to $$v_2$$ and $$I(z_1;v_2)$$ is view-consistent information shared by $$z_1$$ and $$v_2$$.

Therefore we need to optimize both $$\min I(v_1;z_1 \vert v_2)$$ and $$\max I(z_1;v_2)$$ for consistency of $$z_1$$.

We can use the following equation to approximate the upper bound of $$I(v_1;z_1 \vert v_2)$$.

$$ 
\begin{aligned}
\underset{\theta, \phi}{\min} \mathbb{E}_{v_1,v_2 \sim E_{\theta}(v \vert x)} [\mathbb{E}_{z_1,z_2 \sim E_{\phi}(z \vert v)}[KL(p(y \vert z_1) \parallel p(y \vert z_2))]],
\end{aligned}
$$

Similarly, we can use the following equation for $$I(v_2;z_2 \vert v_1)$$.

$$ 
\begin{aligned}
\underset{\theta, \phi}{\min} \mathbb{E}_{v_1,v_2 \sim E_{\theta}(v \vert x)} [\mathbb{E}_{z_1,z_2 \sim E_{\phi}(z \vert v)}[KL(p(y \vert z_2) \parallel p(y \vert z_1))]],
\end{aligned}
$$

By combining the 2 equations, we obtain the following (symmetric) loss function to minimize view-specific information for both $$z_1$$ and $$z_2$$:

$$ 
\begin{aligned}
\mathcal{L}_{VML} = \underset{\theta, \phi}{\min} \mathbb{E}_{v_1,v_2 \sim E_{\theta}(v \vert x)} [\mathbb{E}_{z_1,z_2 \sim E_{\phi}(z \vert v)}[JSD(p(y \vert z_1) \parallel p(y \vert z_2))]].
\end{aligned}
$$

### 2.2. VCD

Now, let us take a look at $$I(z_1;v_2)$$ of $$I(v_1;z_1) = I(v_1;z_1 \vert v_2) + I(z_1;v_2)$$ in the chapter 2.1.

$$ 
\begin{aligned}
I(z_1;v_2) = I(v_2;z1 \vert y) + I(z_1;y).
\end{aligned}
$$

This implies that

$$ 
\begin{aligned}
I(v_1;z_1) = I(v_1;z_1 \vert v_2) + I(v_2;z1 \vert y) + I(z_1;y).
\end{aligned}
$$

From the equation, we can find out that even view-consistent information ($$I(z_1;v_2)$$) would includes superflous one ($$I(v_2;z1 \vert y)$$).

By the Theorem 2 of the paper, $$z_1$$ and $$z_2$$ are view-consistent if the following conditions are satisfied.

$$ 
\begin{aligned}
I(v_1;z_1 \vert v_2) + I(v_2;z1 \vert y) = 0, \newline
I(v_2;z_2 \vert v_1) + I(v_1;z2 \vert y) = 0.
\end{aligned}
$$

According to the Theorems and Corollary, we can use the following loss function to minimize superflous information from $$z_1$$, and vice versa for $$z_2$$:

$$ 
\begin{aligned}
\mathcal{L}_{VCD} = \underset{\theta, \phi}{\min} \mathbb{E}_{v_1,v_2 \sim E_{\theta}(v \vert x)} [\mathbb{E}_{z_1,z_2 \sim E_{\phi}(z \vert v)}[KL(p(y \vert v_2) \parallel p(y \vert z_1))]].
\end{aligned}
$$

### 2.3. MI losses for the experiments

For 2 type view data, and for each 'encoder + IB' block with 1-type input or 2-type input, 1-type input blocks use $$\mathcal{L}_{VSD}$$ and 2-type input block uses both $$\mathcal{L}_{VML}$$ and $$\mathcal{L}_{VCD}$$.

Thus, for the overall loss, the MI loss is given as:

$$ 
\begin{aligned}
\lambda \cdot (\mathcal{L}_{VSD} + \mathcal{L}_{VML} + \mathcal{L}_{VCD}),
\end{aligned}
$$

with the coefficient $$\lambda$$.



## Reference

1. Tian, Xudong, et al. "Farewell to mutual information: Variational distillation for cross-modal person re-identification." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

2. Federici, Marco, et al. "Learning robust representations via multi-view information bottleneck." arXiv preprint arXiv:2002.07017 (2020).



## Reviews of the Mainstream of the Information Bottleneck Framework

[The Information Bottleneck Method (1999)](https://disin7c9.github.io/review/2023-08-16-The-Information-Bottleneck-Method)

[Opening the Black Box of DNNs via Information (2017)](https://disin7c9.github.io/review/2023-08-18-Opening-the-Black-Box-of-Deep-Neural-Networks-via-Information)