---
layout: post
title: "Simple Activity Trakcer"
description: A simple Python webcam app for tracking position and evaluating the activity level of a subject based on YOLOv5
author: "Dongin Sin"
date: 2025-01-22
categories: works
tags: [simple, python, object detection, opencv, yolo]
---

# Simple Activity Trakcer

## Development log

This is a log of **simple** coding work I did while working at a startup.

The CTO and I cooperated to create a python webcam app that detects an object using YOLOv5 and estimates its activity from input video.

I approximated the activity by determining whether the center coordinates of an object in video exceed a given criterion.


![figure1](/assets/img/works/SimpleActivityTracker/motion_yolo_2.png)
figure1: Usage example when terminated
{:.figure}

Source code: [https://github.com/disin7c9/SimpleActivityTracker](https://github.com/disin7c9/SimpleActivityTracker)
