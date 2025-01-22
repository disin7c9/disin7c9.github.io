---
layout: post
title: "Simple Image Similarity"
description: A simple Python app for retrieving the closest image by calculating image similarity
author: "Dongin Sin"
date: 2025-01-22
categories: works
tags: [simple, python, image similarity, opencv, imagehash]
---

# Simple Image Similarity

## Development log

This is a log of **simple** coding work I did while working at a startup.

The CTO requested me to create an python app that calculates image similarity and retrieves the closest image from a given image set for an input.

I used ORB algorithm from OpenCV and pHash from ImageHash; both handle images in grayscale.


![figure1](/assets/img/works/SimpleImageSimilarity/image_similarity.png)
figure1: Usage example
{:.figure}

Source code: [https://github.com/disin7c9/SimpleImageSimilarity](https://github.com/disin7c9/SimpleImageSimilarity)

Sample image set source: [https://www.zeldadungeon.net/wiki/Gallery:Tears_of_the_Kingdom_Characters](https://www.zeldadungeon.net/wiki/Gallery:Tears_of_the_Kingdom_Characters)
