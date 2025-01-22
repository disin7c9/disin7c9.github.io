---
layout: post
title: "Diffusion Drawing"
description: A Python drawing app with Stable Diffusion Webui Forge using Docker and a web server like an AWS instance
author: "Dongin Sin"
date: 2025-01-22
categories: works
tags: [python, diffusion model, Stable Diffusion, webui-forge, Docker, AWS, PyQt]
---

# Diffusion Drawing

## Development log

This is a log of coding work I did while working at a startup.

The CTO provided me with some references and requested that I create a python app and a server-side image generation method.

I chose Docker, AWS instance service, and Stable Diffusion WebUI-forge to fulfill the request.

This python app itself includes basic drawing functionality, prompt inputs, and options like module, model, and hyperparameter adjustments, but it does not have an image generation feature. It can only send generation requests to the server and receive the results.

For server-side configurations like Docker settings, refer to the instructions directory on the Github page.


![Example1](/assets/img/works/DD/DiffusionDrawing_1.png)
figure1: Usage example
{:.figure}

![Example1-1](/assets/img/works/DD/DiffusionDrawing_2.png)
figure2: The example with terminals
{:.figure}

![Example2](/assets/img/works/DD/DiffusionDrawing_4.png)
figure3: Another example
{:.figure}

![Example2-1](/assets/img/works/DD/DiffusionDrawing_5.png)
figure4: The example with terminals
{:.figure}


Source code: [https://github.com/disin7c9/DiffusionDrawing](https://github.com/disin7c9/DiffusionDrawing)
