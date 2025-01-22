---
layout: post
title: "A Character Created Using ChatGPT And ElevenLabs"
description: A text- and voice-interactable character that embodies a fictional character's personality and voice, created using ChatGPT assistants feature and ElevenLabs fine-tuned TTS
author: "Dongin Sin"
date: 2025-01-22
categories: works
tags: [python, TTS, STT, ChatGPT, ElevenLabs, fine-tuning, Frieren]
---

# A Character Created Using ChatGPT And ElevenLabs

## Development log

This is a log of coding work I did while working at a startup.

The CTO provided me with some references and requested that I try to create a python based messenger app that allows both text typing and voice input interactions with a fictional character, Frieren, from a Japanese animation.

I used ChatGPT's assistants API feature.
Additionally, I chose Google speech recognition and ElevenLabs instead of OpenAI whisper and XTTS libraries for faster dev trials.


![figure1](/assets/img/works/GPT-ElevenLabs-Charater/GPTChar_vtube.png)
figure1: A conversation example with VTube studio and VB-Cable (live2d model source: https://kyoki.booth.pm/items/5323958)


{:.figure}

![figure2](/assets/img/works/GPT-ElevenLabs-Charater/GPTChar_messenger.png)
figure2: A messenger app example based on KivyMD
{:.figure}

Source code: [https://github.com/disin7c9/GPT-Elevenlabs-Character](https://github.com/disin7c9/GPT-Elevenlabs-Character)


## Reference

[https://sesang06.tistory.com/216](https://sesang06.tistory.com/216)