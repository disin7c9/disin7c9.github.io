---
layout: post
title: "Opening the Black Box of Deep Neural Networks via Information"
description: review the information bottleneck method and the following-up papers to take a look IB framework's mainstream.
---


Even the brilliant success of deep neural networks, there is no comprehensive theoretical base of deep learning until 2016.
Tishby and Zaslavsky proposed to analyze DNN via the information plane in 2015.
They also argued that the object of NN is to optimize the Information Bottleneck (IB) tradeoff between compression and prediction for each layer.


As a follow-up study, in this paper, Schwartz-Ziv and Tishby present the following main results:

1. Most of training epochs in standard DL are spent on compression of the input to efficient representation and not on fitting the training labels.

2. The phase shift from Empirical Risk Minimization to representation compression is begin when the training loss is becomes small and the Stochastic Gradient Decent (SGD) epochs changes from a fast drift to smaller training loss into a stochastic relaxaion(=random diffusion).

3. The converged layers lie on or very close to the IB theoretical bound (=the convex curve of $$ (I_X, I_Y) $$ for each $$ \beta $$), and the maps $$ p(T_i \vert X) $$ and $$ p(Y \vert T_i) $$ satisfy the IB self-consistent equations.

4. The training time dramatically is reduced when adding more hidden layers. 
Thus the main advantage of the hidden layer is computational. 
This phenomenon can be explained by the reduced relaxation time, as this it scales super-linearly with the information compression from the previous layer.

5. As we expect critical slowing down of stochastic relaxation near phase transitions on the IB curve, we expect the hidden layers to converge to such critical points.


# Main

The optimization of DNN by SGD has the following 2 different and distinct learning phases:

- Empirical Risk Minimization (ERM)
- representation compression

and this phenomenon was first revealed in this paper. These 2 phases are distingushed by pretty different Signal-to-Noise Ratios (SNR) of the stochastic gradients in each layer.
SNR is the log difference between the normalized mean and the standard deviation (STD) of gradients. In the ERM phase the means are much bigger than the STDs, and in the representation compression phase vice versa.


- Notation

-- $$ X $$: input random variable

-- $$ Y $$: desired output variable relevant to $$ X $$

-- $$ T_i $$: the compressed representative of $$ X $$ through $$i^{th}$$ hidden layer of the DNN

-- $$ P(T \vert X) $$: encoder distribution

-- $$ P(Y \vert T) $$: decoder distribution


## 1. Information Theory of Deep Learning

In DNN training, we are interested in representations $$ T(X) $$ of $$ X $$ for prediction $$ Y $$.
Furthermore we want efficient and generalized representative learning from empirical samples of the unknown joint distribution $$ P(X,Y) $$.


### 1.1. Mutual Information

For DNNs, there are 2 important properties of Mutual Information (MI).

- 1st. invariance to invertible trasformations: 

$$
\begin{aligned}
I(X;Y) = I(\psi(X); \phi(Y))
\end{aligned}
$$

for any invertible functions $$ \phi $$ and $$ \psi $$.

- 2nd. Data Processing Inequality (DPI): 

$$
\begin{aligned}
I(X;Y) \geq I(X;Z)
\end{aligned}
$$

for any 3 variables that form a Markov chain $$ X \rightarrow Y \rightarrow Z $$.


### 1.2. The Information Plane

Given $$ P(X,Y) $$, $$ T$$ is uniquely mapped into a point in the information plane with coordinates $$ (I_X, I_Y) $$.
Consequently for all K hidden layer $$ {T_i} $$, they are mapped to K monotonic connected points in the plane (henceforth a unique information path) which satisfies the following DPI chains:

$$
\begin{aligned}
I(X;Y) \geq I(T_1;Y) \geq I(T_2;Y) \geq \dots \geq I(T_k;Y) \geq I(\hat{Y};Y) \newline
H(X) \geq I(X;T_1) \geq I(X;T_2) \geq \dots \geq I(X;T_k) \geq I(X;\hat{Y}).
\end{aligned}
$$

each information path in the plane corresponds to many different DNN's with probably very different architectures since the first property of 1.1.


### 1.3. The Information Bottleneck optimal representations

In the paper's context, **sufficient statistics** $$ S(X) $$ are maps or partitions of X that contain all the information of Y; i.e., $$ I(S(X);Y) = I(X;Y) $$.
Using the DPI, the most simple sufficient statistics $$ T(X) $$ can be denoted by a constrained optimization problem:

$$
\begin{aligned}
T(X) = \underset{S(X):I(S(X);Y)=I(X;Y)}{\arg\min}I(S(X);X)
\end{aligned}
$$

These $$T(X)$$, minimal sufficient statistics, are hard to search out.
In 1999, Tishby et al. proposed *the Information Bottleneck* which provide a computational framework for finding approximate minimal sufficient statistics, or the optimal tradeoff between compression of $$X$$ and prediction $$Y$$.


### 1.4. The crucial role of noise

If we don't know the structure or topology of $$X$$, even for a binary $$Y$$, it is impossible to distingush low complexity classes from highly complex classes only with MI.

However we can solve this obstacle by adding small noise to the input patterns with sigmoid function.


### 1.5. Visualizing DNNs in the Information Plane

As proposed by Tishby and Zaslavsky in 2015, the authors researched the infomation path of DNNs in the information plane.
For the study, it is nessesary to know the underlying joint distribution $$ P(X,Y) $$ and be able to calculate $$ P(T \vert X) $$ and $$ P(Y \vert T) $$ directly.

It is possible to compare and visualize different NN architectures in terms of efficient preserving the relevant information by using ordered pair $$ (I_X, I_Y) $$ in the information plane.
The paper covers the following topics:

1. The SGD layer dynamics in the *Information Plane*. &nbsp; - **(A)**

2. The effect of the training sample size on the layers. &nbsp; - **(B)**

3. What is the benefit of the hidden layers?  &nbsp; - **(C)**

4. What is the final location of the hidden layers?  &nbsp; - **(D)**

5. Do the hidden layers form optimal IB representations?  &nbsp; - **(E)**


## 2. Numerical Experiments and Results

### 2.1. Experimental Setup

(skip)


### 2.2. Estimating the MI of the layers

(skip)


### 2.3. The dynamics of the training by SGD

(ploting, skip)


### 2.4. The two optimization phase in the Information Plane &nbsp; - (A), (D)

At the begining of training by SGD, the NN increases $$ I(T_i;Y) $$ while preserving the DPI order regardless of the amount of data.
This period is the ERM phase.
Then, over relatively much longer training epochs, $$ I(X;T_i) $$ decreases and NN removes irrelevant information until convergence.
This period is the *representation compression phase*.

Unlike the ERM phase, the compression phase is probably surprising and unexpected phenomenon.
There was no explicit regularization in the experiment and the same phase shift was observed for other problems.
This 2-phases learning seems to be common in training DNNs by SGD.

In particular, the ERM phase of each NN was similar no matter the sizes of training sample, but the compression phase was not.
$$ I(T_i;Y) $$ decreased when the sample size was small and conversely increased when it was large.
It may looks like overfitting to the small size sample.
However this overfitting is largely due to the loss of relevant information accompanying the simplification of the representations of the layers during the compression phase.

Besides, it is noteworthy that each layer of NNs with random initial weights is optimized along very similar information path and eventually clusters around a certain point in the information plane.


### 2.5. The drift and diffusion phases of SGD optimization &nbsp; - (A), (D)

Let us check the behavior of stochastic gradients per epoch for each phase.
Entire training is clearly divided into 2 distinct phases.
The first is a drift phase, where the normalized means of gradients are relatively bigger than their own STDs.
The second is a diffusion phase, where the opposite of the previous phase, and the gradients behave as Gaussian noise with very small means.
Each phase corresponds to ERM and compression phase respectively.

Such dynamic phase shift simultaneously occurs in each layer. 
For every layer, $$ I_Y $$ increases in the drift phase since it reduces the loss.
On the other hand, random noise is added to weights in the diffusion phase under $$ I_Y $$ constraint.
Consequently, it maximizes $$ H(X \vert T_i) $$(i.e., minimize $$ I(X;T_i) $$).
This entropy maximization by additive noise is known as stochastic relaxation.

However, it is uncertain why the different depth layers converge to distinct points in the information plane.

Another interesting point is randomized characterictic of the converged weights after compression by diffusion phase.
There was no indication of weights vanishing or norm decreases near the convergence.
Furthermore, the correlations between in-weights of different neurons in the same layer was very small.
This indicates that there are numerous different optimized networks, and trying to interpret single neuron in such networks would be meaningless.


### 2.6. The computational benefit of the hidden layers &nbsp; - (C)

Here are the results from another experiment.

1. *Adding hidden layers dramatically reduces the number of training epochs for good generalization.*

2. *The compression phase of each layer is shorter when it starts from a previous compressed layer.*

3. *The compression is faster for the deeper layers.*

4. *Even wide hidden layers eventually compress in the diffusion phase. Adding extra width does not help.*
(probably it means that overcompleteness is uselss.)

### 2.7. The computational benefits of layered diffusion &nbsp; - (C)

(could not summarize due to lack of background knowledge; supplement it later.)


### 2.8. Convergence to the layers to the Information Bottleneck bound &nbsp; - (E)

To evaluate the IB optimality of each layer the authors tested whether the converged layer satisfied **self-consistency** for some $$\beta$$.

For the evaluation, they calculated optimal IB encoder $$ p_{i,\beta}^{IB}(t \vert x)$$ from the $$i^{th}$$ layer decoder $$ p_i(y \vert t)$$.
Then, they obtained the optimal $$ \beta_i $$ for each layer through minimizing the averaged Kullback-Leibler divergence between the IB and the layer's encoders.

$$ (I_X^i, I_Y^i) $$ and the IB information curve show that each converged layer impressively close to the theoretical IB limit ($$ R(D) $$), where the slope of the curve $$\beta^{-1}$$ matches to the estimated optimal $$\beta_i^*$$


### 2.9. Evolution of the layers with training sample size &nbsp; - (B)

Each layer conveges to specific point on the information curve with finite sample.

In the shallower layers, the training sample size hardly effect the information, since even random weights preserve most of the MI $$I_X$$, $$I_Y$$.
On the other hand, in the deeper layers, the NN learns to preserve relevant information and to compress the irrelevant information.
With larger training samples more details on $$X$$ become relevant for $$Y$$, which leads to a shift to higher $$I_X$$ in the middle layers.



## Reference

- Shwartz-Ziv, Ravid, and Naftali Tishby. "Opening the black box of deep neural networks via information." arXiv preprint arXiv:1703.00810 (2017).



## Reviews of the Mainstream of the Information Bottleneck Framework

[The Information Bottleneck Method (1999)](https://disin7c9.github.io/review/2023-08-16-The-Information-Bottleneck-Method)

[Opening the Black Box of DNNs via Information (2017)](https://disin7c9.github.io/review/2023-08-18-Opening-the-Black-Box-of-Deep-Neural-Networks-via-Information)

[Farewell to Mutual Information (2021)](https://disin7c9.github.io/review/2023-08-19-Farewell-to-Mutual-Information)