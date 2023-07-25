---
layout: post
title: "Generative Adversarial Nets: GAN & DCGAN"
description: review GAN and pytorch tutorial DCGAN
---

As of 2020, the most popular approaches to generative modeling are probably GANs, variational autoencoders, and fully-visible belief nets.

In this review, we will look at the original GAN and one of its variants, DCGAN.


## 1. Notations

- Probability density functions

$$ p_g $$ : the generator's distribution over data $$ x $$.

$$ p_{data} $$ : the data generating distribution.

$$ p_z $$ : input noise variables' distribution.

- Models

$$ G(z;\theta_g) $$ : the multilayer perceptron generator $$ G $$ with parameters $$ \theta_g $$. $$ G $$ maps $$ z $$ to $$ x $$.

$$ D(x;\theta_d) $$ : the multilayer perceptron discriminator $$ D $$ with parameters $$ \theta_d $$. $$ D(x) $$ represents the probability that $$ x $$ came from the data.


## 2. Value Function

D and G play 2-player minmax game with the value function $$ V(D,G) $$.

$$
\begin{gathered}
  \underset{G}{\min}\underset{D}{\max}V(D,G) \\ 
  \text{with} \\  
  V(D,G) = \mathbb{E}_{x{\sim}p_{data}(x)}[logD(x)] + \mathbb{E}_{z{\sim}p_{z}(z)}[log(1-D(x))].
\end{gathered}
$$


The original paper offered 2 versions of the loss function for the generator.

### 2.1. minimax GAN (M-GAN, theoretical)

M-GAN defined a cost $$ J^{(G)} = -J^{(D)} $$.

For real and fake training data

$$
\begin{cases}
  x, y(=1) & x \text{ is real} \newline
  x, y(=0) & x \text{ is fake},
\end{cases}
$$

- Training $$D$$:
$$
\underset{D}{\max}V(D,G) = \underset{D}{\max} \left( \mathbb{E}_{x{\sim}p_{data}(x)}[logD(x)] + \mathbb{E}_{z{\sim}p_{z}(z)}[log(1-D(x))] \right)
$$

- Training $$G$$:
$$
\underset{G}{\min}V(D,G) = \underset{G}{\min} \left( \mathbb{E}_{z{\sim}p_{z}(z)}[log(1-D(x))] \right).
$$

### 2.2. Non-Saturating GAN (NS-GAN, prevent gradient saturation)

When $$G$$ is poor, especially in the early step of training, $$D$$ can easily reject data from $$z$$. In this case, $$ log(1-D(x)) $$ saturates ($$ \nabla{V(D,G)} \approx 0 $$), because what ever $$z$$ is, $$ D(G(z)) \approx 0 $$.

Therefore NS-GAN flips the labels when the generator is being trained.

- Training $$D$$:

For real and fake training data 

$$
\begin{cases}
  x, y(=1) & x \text{ is real} \newline
  x, y(=0) & x \text{ is fake},
\end{cases}
$$

objective: $$ \underset{D}{\max}V(D,G) = \underset{D}{\max} \left( \mathbb{E}_{x{\sim}p_{data}(x)}[logD(x)] + \mathbb{E}_{z{\sim}p_{z}(z)}[log(1-D(x))] \right) $$.

- Training $$G$$:

for input noises and labels 

$$ x, y(=1)\quad x \text{ is fake}, $$

objective: $$ \underset{G}{\max}V(D,G) = \underset{G}{\max} \left( \mathbb{E}_{z{\sim}p_{z}(z)}[logD(x)] \right). $$


## 3. Theoretical Results

The generator defines pdf $$p_g$$ implicitly. ("GANs are implicit models that infer the probability distribution p(x) without necessarily representing the density function explicitly.")

Assumption: $$D$$ and $$G$$ have no parametric restrictions.

### 3.1. Global Optimality of $$ p_g = p_{data} $$.

#### 3.1.1 proposition 1. (optimal $$D$$)
For fixed $$G$$, the optimal discriminator $$D$$ is 

$$
D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

-  Note on proof

The training criterion for $$D$$ is to maximize the $$V(G,D)$$.

$$
\begin{gathered}
  V(G,D) = \int_{x}p_{data}(x)logD(x) + p_g(x)log(1-D(x))dx \\

  \text{and} \\
  \\

  f(y) := a\log(y) + b\log(1-y). \\
  \forall \ (a,b) \in \mathbb{R}^2/\{0,0\}, \ y \in [0,1], \\ 
  \underset{y}{argmax} f(y) = \frac{a}{a+b}
\end{gathered}
$$

#### 3.1.2 Theorem 1. ($$\exists!$$ optimal $$G$$)

The global minimum of the virtual training criterion $$C(G)$$ is achieved iff $$p_g=p_{data}$$. At the point, $$C(G)$$ achieves the value $$-\log4$$.

- Note on proof

$$p_{data}$$ is apparent optimal value of $$p_g$$.

$$p_g=p_{data} \Rightarrow D^*_G(x)=\frac{1}{2} \Rightarrow V(D^*_G,G)=C(G)=-\log4$$

Therefore the difference of the optimal $$C(G)$$ with an arbitrary $$p_g$$ and the optimal $$C(G)$$ with $$p_g=p_{data}$$ is

$$
\begin{aligned}
  C(G) - (\mathbb{E}_{x{\sim}p_{data}}[-log2] + \mathbb{E}_{x{\sim}p_g}[-log2)])
  &= \int_{x}p_{data}log\frac{p_{data}}{p_{data}+p_g}dx + \int_{x}p_glog\frac{p_g}{p_{data}+p_g}dx -(-log4) \\
  &= - \int_{x}p_{data}log\frac{p_{data}+p_g}{2p_{data}}dx - \int_{x}p_glog\frac{p_{data}+p_g}{2p_g}dx \\
  &= KL(p_{data} \parallel \frac{p_{data}+p_g}{2}) + KL(p_g \parallel \frac{p_{data}+p_g}{2}) \\ 
  &= 2JSD(p_{data} \parallel p_g)
\end{aligned}
$$

### 3.2. Convergence of the Algorithm

#### Proposition 2.

If $$G$$ and $$D$$ have enough capacity, and at each step of the Algorithm, the discriminator is allowed to reach its optimum given $$G$$, and $$p_g$$ is updated so as to improve the criterion

$$\mathbb{E}_{x{\sim}p_{data}}[logD^*_G(x)] + \mathbb{E}_{x{\sim}p_g}[log(1-D^*_G(x))]$$

- Note

-- the point where the maximum is attained: $$\beta = \underset{\alpha \in A}{argsup} f_{\alpha}(x)$$

-- the subderivatives of a supremum of convex functions: $$\partial f$$

-- the derivative of the funcion at the point: $$\partial f_{\beta}(x)$$

-- $$\alpha : D$$

-- $$\beta : D^*_G$$


## 4. PyTorch Tutorials: DCGAN

Training techniques of DCGAN

- update D Network

-- step1: train with all real batch

-- step2: train with all fake batch

- update G Network

-- Use NS-GAN loss


## 5. References

- Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

- Goodfellow, Ian, et al. "Generative adversarial networks." Communications of the ACM 63.11 (2020): 139-144.

- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial