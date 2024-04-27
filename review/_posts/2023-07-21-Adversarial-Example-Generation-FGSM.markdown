---
layout: post
title: "Adversarial Example Generation: FGSM"
description: review pytorch tutorial FGSM
---

Basically the key point of adversarial attacks is to cause a model to malfunction by adding the least amount of perturbation to the input data.

There are several types of assumptions of the attacker's goals, in particular, misclassification and sourse/target misclassification.

- A goal of (simple) misclassification: the adversary wants images are classified as wrong target class.
- A goal of sourse/target misclassification: the adversary wants images of a specific source class are classified as another specific target class.

Also there are several kinds of assumptions of the attacker's knowledge, in particular, white box and black box.

- White box: Assumption that the attacker has full knowledge and access to the model, including architecture, inputs, outputs, and parameters.

- Black box: Assumption that the attacker only has access to the inputs and outputs.


## Tutorial Setting

Use the Fast Gradient Sign Attack (FGSM).

data: MNIST

model: LeNet

purpose: simple misclassification


## Mathematical Concept

Perturbed image $$ x = x + \epsilon \ast sign\left(\nabla_{x}J(\theta,x,y)\right) $$. Here \\
\\
$$
\begin{aligned}
 \nabla_{x}J(\theta,x,y) =& \begin{bmatrix}
			\frac{\partial J}{\partial x_{11}} & \cdots & \frac{\partial J}{\partial x_{1n}}\\
		 	\vdots		 	      & \ddots & \vdots			\\
		  	\frac{\partial J}{\partial x_{m1}} & \cdots & \frac{\partial J}{\partial x_{mn}}
			\end{bmatrix}
\end{aligned}.
$$


Since $$\nabla_{x}J(\theta,x,y)$$ is partial derivates matrix of loss $$J$$ with respect to the input $$x$$, $$J$$ increases for each increment of $$ x_{ij} $$ in the direction of $$\frac{\partial J}{\partial x_{ij}}$$.


# References

- https://pytorch.org/tutorials/beginner/fgsm_tutorial