---
layout: post
title: "Try Fine-Tuning LLMs at Home"
description: practice fine-tuning with LLaMA.
author: "Dongin Sin"
date: 2024-11-21
categories: practice
tags: [fine-tuning, LLMs, AI, Unsloth, Continued Pre-Training, LoRA]
---


![figure1](/assets/img/practice/try_fine-tuning_LLMs/cute_lama_by_Flux.png)
figure1: A cute lama, generated by Flux 1.1 pro
<!-- A chibi cute lego lama peacefully wandering in the grasslands of the Andes, with bold "Let's fine-tune LLMs!" text in a sans-serif font above it -->
{:.figure}


# Intro

In this post, we are going to briefly understand and practice LLMs fine-tuning.

What you need: 
 - PC
 - one of the following AI accelerators:
   - GPU, probably RTX 20 series or higher
   - Cloud computing service like Colab or AWS instance
 - Python, recommended a virtual environment


# Index

1. What is fine-tuning? why do we need to do that?
2. What should we know before try fine-tuning?
  - LoRA
  - Quantization
  - Batch size and gradient accumulation
3. Try fine-tuning
  - What do we do?
  - Fine-tuning code
  - Loss graphs
  - Inference test
4. Conclusion
5. Other informative posts
6. References


# 1. What is fine-tuning? why do we need to do that?

Fine-tuning is not the only concept in language models area, but let us concentrate on this domain.

Companies like OpenAI, META, Google and Microsoft have released their own large language models.
Such models are quite intelligent and can handle general tasks like Q&A, summary, classification, etc. in common sense.

However, even those state-of-the-art models are not perfect to some specific tasks or domains where they have never experienced.
For example, if one model was not trained on medical data, it cannot be used as medical assistant.
Also, in case of LLaMA 3, the large language model META AI, it only supports English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai officially.
Then we can easily expect that LLaMA is not good at Japanese or Korean as much as officially supported languages.
Beside, for image+text applications, English is the only language supported by multimodal LLaMA.

But training a LLM from scratch is very very complicated and expensive work, and furthermore, some big-tech enterprises released their model as free. 
i.e., by choosing pre-trained open-source model, individuals or small companies can save their resources.
So what we need to do is just pick a model understanding common knowledges and tune the model for our own tasks.


# 2. What should we know before try fine-tuning?

You can skip this part, but before starting fine-tuining immediately, let us understand the following few keypoints:
 - LoRA and its rank and alpha options
 - Quantizaion
 - Batch and gradient accumulation

Also my previous posts about
[Transformer](https://disin7c9.github.io/review/2023-09-21-Attention-is-All-You-Need/) 
and 
[large language models](https://disin7c9.github.io/summary/2023-12-21-Large-Language-Models-before-ChatGPT-3.5/)
would be helpful.


## LoRA

![figure2](/assets/img/practice/try_fine-tuning_LLMs/lora.png)
figure2: LoRA mechanism (source: *"Lora: Low-rank adaptation of large language models. (2021)"*)
{:.figure}

Fine-tuning meant that training all the parameters to another dataset after pre-training.
However *“Scaling Laws for Neural Language Models (2020)”* found **the scaling law** that the model scale is the most important feature in performance than others like data amount, training steps, and model shape.
So it is hard to train all the weights of LLMs because the sizes are usually bigger than 1 billion.

In 2021, Hu, Edward J., et al. introduced LoRA technique and it made an impact. [^1] 
This technique freezes the original parameters of a LLM and adds weight matrices as adapter.
In detail, for an original weights matrix $$W$$, LoRA method attachs additional trainable matrix $$\Delta W$$ with **rank** $$r$$ and **scaling factor** $$\alpha$$ like 

$$
\begin{gathered}
W'x = (W + \frac{\alpha}{r}\Delta W) x, \newline
\text{with} \quad \Delta W = B \cdot A \quad \text{for} \quad W, \Delta W \in \mathbb{R}^{(d,d)}, B \in \mathbb{R}^{(d,r)}, A \in \mathbb{R}^{(r,d)} \quad \text{where} \quad r << d,
\end{gathered}
$$

$$x$$ is an input representative vector.

Also, the low-rank matrix $$\Delta W = BA$$ has rank at most $$r$$, i.e., there are at most $$r$$ linearly independent vectors.
It can be factorized as a sum of $$r(<<d)$$ rank-1 matrices using outer products of vectors.
(This is conceptually similar to how low-rank approximations are used in SVD, though LoRA uses directly learnable matrices $$B$$ and $$A$$ rather than decomposing $$W$$.)

So we need to optimize $$r$$ and $$\alpha$$ since those are related to learning capacity and regulator of the LoRA adapter.

The benefits of LoRA are came from:
1. **Efficiency**, because we need to only train and use $$ 2 \times r \times d $$ weights instead of $$ d^2 $$.
2. **Flexibility**, because we need to only save and shift the LoRA weights adapter.


## Quantization

![figure3](/assets/img/practice/try_fine-tuning_LLMs/quant.png)
figure3: Quantization method in *LLM.int8() paper*
{:.figure}

Most of LLMs parameters are saved as 'standard single-precision floating-point form' a.k.a float32.
Each number expressed in the float32 data type requires 32 bit of information quantities.
However, because of the large scale of language models, this causes computationally intensive training and slow inference.
I don't know the latest research about quantization methods, but the concept of quantization is similar to clustering.
Given parametric data, we group them into several points by mapping floating-point values to discrete levels (e.g., integers).
We may use scaling to transform the grouped floating numbers into a smaller integer representation, while being aware of outliers. [^2]

Subsequently, the model parameters are downsized from float32 to int8, int4, or even more [compact](https://github.com/microsoft/bitnet) forms, enabling faster inference and reduced hardware requirements. [^3], [^4]


## Batch and Gradient Accumulation

![figure4](/assets/img/practice/try_fine-tuning_LLMs/grad_accm.png)
figure4: Simple batch training and the corresponding gradient accumulation (source: https://unsloth.ai/blog/gradient)
{:.figure}

Batch size, precisely mini-batch size, is one of practical options in fine-tuning.
Larger batch size tends to stablize training and boost convergence, but also takes a lot of GPU memory.
Fortunately, Hugging Face transformers library with PEFT or Unsloth supports gradient accumulation feature.
If we use gradient accumulation, the optimizer does not update the weights immediately right after loss calculation for each batch.
Instead, it stacks loss values until reaching given **gradient_accumulation_steps**, averages the loss sum 'properly', and then update the weights.
So this **batch_size** $$\times$$ **gradient_accumulation_steps** equals to actual larger batch size.

Therefore, this trick enables to mimic larger and more effective batch training while reducing VRAM usage. [^5]



# 3. Try fine-tuning

There is no only one way to fine-tune LLMs.

You can use web-based services such as OpenAI or Kaggle without a bunch of complicated codelines.

You can also do it **free** by downloading models to your computer with Hugging Face API.

In case of Hugging Face,
There are also some various packages and open-sources compatible to Hugging Face: PEFT, TRL, Unsloth, Qwen-VL etc., and each library is not exclusive to others.

[Unsloth](https://unsloth.ai/) is a novel open-source Python library for efficent fine-tuning of LLMs.
The brothers, who developed the library, introduce that Unsloth is faster and more memory-efficient than the original implements of Hugging Face.
Also they manually scripted autograd instead of torch autograd and rewrote OpenAI Triton kernels.
Although training on multi-GPU is not free, but it is no matter for individuals or students.

## What do we do?

If you want to fine-tune an open-source LLM on your own task, it might be insufficient to train it immediately on your task specific dataset.
There are several reasons:
1. Most LLMs are mainly exposed to English data and may not have common sense about the other cultures.
2. If you want to prevent overfitting, you must not reuse the data too much time during fine-tuning. In this case, Scaling up the dataset would be helpful, but it is hard to obtain such well-refined dataset.

To handle this situation, there is a solution: **Continued Pre-Training**.

Then what is continued pre-training?
For example, if I want to make a Korean counseling chatbot from LLaMA, first I train it dataset with full of Korean context and culture like Korean wikipedia pages, then fine-tune it toward my own counseling-task dataset.

The other main ideas distinct to ordinary LoRA fine-tuning are well explained in [the official blog post of Unsloth](https://unsloth.ai/blog/contpretraining), but here is the summary:
 - Train input and output embeddings too (embed_tokens and lm_head), but use different learning rate.
 - Use rank stablized LoRA. [^6]


Therefore I chose Unsloth and LLaMA to try fine-tuning in this post.
[Visit my github repo for the full version code.](https://github.com/disin7c9/LLaMA_3_fine-tuning_example)


## Fine-tuning code

I slightly reformed the [original code](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=idAEIeSQ3xdS) on my purpose, but it is only for validation, inference, GGUF, saving and loading. The mainstream is equivalent to the original.

With batch_size = 2, VRAM usage was less than 10GB, probably 7~9 during fine-tuning.
This reduced memory consuming was really beneficial.

### Build environment

Install deep learning packages.

~~~py
!pip install "torch==2.4.0" "xformers==0.0.27.post2" tensorboard pillow torchvision accelerate huggingface_hub transformers datasets accelerate unsloth
~~~

### Import libraries

~~~py
import torch
from transformers import (
    AutoTokenizer,
    TextStreamer,
)
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    unsloth_train,
    UnslothTrainer, 
    UnslothTrainingArguments,
)
from datasets import load_dataset
~~~

### Set important hyperparameters

I used $$alpha = 2 \times rank$$ according to this [paper](https://arxiv.org/pdf/2410.21228v1). [^7]



~~~py
# Important hyperparameters
max_seq_length = 2048 
load_in_4bit = True
BATCH_SIZE = 8
rank = 64
alpha = rank*2
~~~

### Load model for CPT

~~~py
base_model_path = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

# Configure CPT LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = rank, # LoRA rank. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = alpha, # LoRA scaling factor alpha
    lora_dropout = 0.0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = seed,
    use_rslora = True,   # Unsloth supports rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

model.print_trainable_parameters() # the number of trainable weights increase when using "embed_tokens" and "lm_head".
~~~

### Text data formatting for CPT

You can also use **unsloth.get_chat_template** function to get the correct chat template. It supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3 and more. See [example page](https://colab.research.google.com/drive/1z0XJU2FCzDC8oyXa2Nd4jCxylRMI-o0-?usp=sharing#scrollTo=vITh0KVJ10qX).

~~~py
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    # Wikipedia provides a title and an article text.
    wikipedia_prompt = """위키피디아 기사
    ### 제목: {}

    ### 기사:
    {}"""

    titles = examples["title"]
    texts  = examples["text"]
    outputs = []

    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }
~~~

### Prepare data for CPT

You can change the data to your own language or domain relevant dataset.

~~~py
# Load and prepare dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split = "train", )
dataset = dataset.train_test_split(test_size=0.01, shuffle=True, seed=42) # Split dataset into train/validation sets
train_set, val_set = dataset["train"], dataset["test"] 

# Format dataset
train_set = train_set.map(formatting_prompts_func, batched = True,)
print(train_set[0:2])
val_set = val_set.map(formatting_prompts_func, batched = True,)
print(val_set[0:2])
~~~

### Set CPT trainer and run

**unsloth_train(trainer)** codeline handles the gradient accumulation bug.
[Refer to the official post.](https://unsloth.ai/blog/gradient)

~~~py
# Define trainer
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_set,
    eval_dataset=val_set,
    dataset_text_field="text",
    max_seq_length=max_seq_length,

    args = UnslothTrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,  # training batch size
        per_device_eval_batch_size=BATCH_SIZE,  # validation batch size
        gradient_accumulation_steps = 2, # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps

        # Use warmup_ratio and num_train_epochs for longer runs!
        warmup_ratio = 0.1,
        num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        # validation and save
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=5,
        save_safetensors=True,
        
        # # callback
        # load_best_model_at_end=True, # this option is only available when eval_strategy == save_strategy
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,

        # dtype
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear", # try "cosine"
        seed = seed,
        output_dir = "outputs/CPT", # Directory for output files
        report_to="tensorboard", # Reporting tool (e.g., TensorBoard, WandB)
    ),
)

trainer_stats = unsloth_train(trainer)
~~~

### Save CPT LoRA (optional)

~~~py
# Save trained model locally and to Hugging Face Hub
repo_name = "CPT_LoRA_Llama-3.1-8B-Instruct-bnb-4bit_wikipedia-ko"
CPT_path = f"{user_name}/{repo_name}"

# Local
model.save_pretrained(repo_name)
tokenizer.save_pretrained(repo_name)

# Online
model.push_to_hub(
    CPT_path, 
    tokenizer=tokenizer,
    private=True,
    token=HF_write_token,
    save_method = "lora", # You can skip this. Default="merged_16bit" prabably. Also available "merged_4bit".
    )
tokenizer.push_to_hub(
    CPT_path, 
    private=True,
    token=HF_write_token,
    )
~~~

### Load model for Instruction Fine-Tuning (IFT)

~~~py
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = rank, # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",],
    lora_alpha = alpha,
    lora_dropout = 0.0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = seed,
    use_rslora = True,   # Unsloth supports rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

model.load_adapter(CPT_path, adapter_name="CPT")

model.print_trainable_parameters()
~~~

### Text data formatting for IFT
~~~py
# dataset formatting function
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(conversations):
    alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

    ### 지침:
    {}

    ### 응답:
    {}"""
    
    conversations = conversations["conversations"]
    texts = []

    for convo in conversations:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(convo[0]["value"], convo[1]["value"]) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
~~~

### Prepare data for IFT

Again, choose your own task-specific data.

~~~py
alpaca_dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-korean", split = "train")
alpaca_dataset = alpaca_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42) # Split dataset into train/validation sets
alpaca_train_set, alpaca_val_set = alpaca_dataset["train"], alpaca_dataset["test"] 

alpaca_train_set = alpaca_train_set.map(formatting_prompts_func, batched = True,)
print(alpaca_train_set[0:2])
alpaca_val_set = alpaca_val_set.map(formatting_prompts_func, batched = True,)
print(alpaca_val_set[0:2])
~~~

### Set trainer for IFT and run

~~~py
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = alpaca_train_set,
    eval_dataset=alpaca_val_set,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,

    args = UnslothTrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,  # training batch size
        per_device_eval_batch_size=BATCH_SIZE,  # validation batch size
        gradient_accumulation_steps = 4, # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps

        # Use warmup_ratio and num_train_epochs for longer runs!
        warmup_ratio = 0.1,
        num_train_epochs = 2,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        # validation and save
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=10,
        save_safetensors=True,
        
        # # callback
        # load_best_model_at_end=True, # this option is only available when eval_strategy == save_strategy
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,


        # dtype
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear", # try "cosine"
        seed = seed,
        output_dir = "outputs/IFT", # Directory for output files
        report_to="tensorboard", # Reporting tool (e.g., TensorBoard, WandB)
    ),
)

trainer_stats = unsloth_train(trainer)
~~~

### Save IFT LoRA as simple safetensors and GGUF format

~~~py
# Save trained model locally and to Hugging Face Hub as normal and quantized form
repo_name = "IFT_LoRA_Llama-3.1-8B-Instruct-bnb-4bit_wikipedia-ko_alpaca-gpt4-ko"
IFT_path = f"{user_name}/{repo_name}"

# Local
model.save_pretrained(repo_name)
tokenizer.save_pretrained(repo_name)

# Online
model.push_to_hub(
    IFT_path, 
    tokenizer=tokenizer,
    private=True,
    token=HF_write_token,
    save_method = "lora", # also available "merged_16bit", "merged_4bit"
    )
tokenizer.push_to_hub(
    IFT_path, 
    private=True,
    token=HF_write_token,
    )

# GGUF / llama.cpp Conversion
repo_name = "IFT_LoRA_Llama-3.1-8B-Instruct-bnb-4bit_wikipedia-ko_alpaca-gpt4-ko-GGUF"
IFT_GGUF_path = f"{user_name}/{repo_name}"

quantization_method = "q8_0" # or "f16" or "q4_k_m"

model.save_pretrained_gguf(repo_name, tokenizer, quantization_method=quantization_method)
model.push_to_hub_gguf(IFT_GGUF_path, tokenizer, save_method = "lora", quantization_method=quantization_method, private=True, token=HF_write_token)
~~~


## Loss graphs

Since training CPT+IFT definitely take too much time even using Unsloth, 
it's burdensome for me to exhaust so much time and resources on such a practice.
So I skipped CPT and just tried IFT to LLaMA Instruct models 3.2-1B, 3.2-3B and 3.1-8B.

I lost log data of 3.2-1B, so here is the comparison between 3.2-3B and 3.1-8B.

![figure5](/assets/img/practice/try_fine-tuning_LLMs/Llama-3-loss.jpg)
figure5: fine-tuning loss comparision between 3.2-3B and 3.1-8B
{:.figure}

It seems like that the loss values are decreasing well, and trivially, the larger is the better. 
But there are problems.

1. It is doubtful that overfitting has occured after epoch 2.

2. The loss values before the end of 2nd epoch are not sufficiently small enough when compared to the Unsloth official examples of English text data 
[1](https://colab.research.google.com/drive/1z0XJU2FCzDC8oyXa2Nd4jCxylRMI-o0-?usp=sharing#scrollTo=2ejIt2xSNKKp) and [2](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv). Such examples even not trained in 1 full epoch, but only on a few steps.


## Inference test

Here are respective inference results with the same questions:

Q1. What is a famous tall tower in Paris?

Q2. 피보나치 수열을 계속하세요: 1, 1, 2, 3, 5, 8, 
(Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,)

Q3. 지구를 광범위하게 설명하세요.
(Describe the planet Earth extensively.)

Q4. 한국음악은 어떤가요?
(What is Korean music like?)

### 3.2-1B
```
A1.
### Response:
The Eiffel Tower.<|eot_id|>

A2.
### 응답:
  1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1594, 2584, 4181, 6765,

A3.
### 응답:
  - 지구 : "광범위"
  - 광범위 : "광범위"
  - 광범위 : "광범위"
  - 광범위 : "광범위"<|eot_id|>

A4.
### 응답:
  - "한국음악은 어떤가요?"
  - "한국음악은 음악으로 말하는 것"
  - "한국음악은 한국의 음악"
  - "한국음악은 한국의 문화"
  - "한국음악은 한국의 문화를 표현하는 것"

### 예시:
- "한국음악은 어떤가요?"
- "한국음악은 음악으로 말하는 것"
- "한국음악은 한국의 음악"
- "한국음악은 한국의 문화"
- "한국음악은 한국의 문화를 표현하는 것"

###
```

### 3.2-3B
```
A1.
### Response:
The Eiffel Tower is a famous tall tower located in Paris, France. It was built for the 1889 World's Fair and stands 324 meters (1,063 feet) tall. It has become an iconic symbol of the city and one of the most recognizable landmarks in the world.<|eot_id|>

A2.
### 응답:
  1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 

A3.
### 응답:
  지구는 우리โลก입니다. 지구는 지구의 4개의 대륙으로 나뉘어 있습니다. 그 대륙은 아시아, 유럽, 아프리카, 북아메리카, 남아메리카, 오스트레일리아, antarctica, 남極, 인도洋, 태평양, 북대서양, 남대서양, 남서대서양, 남서대서양, 남서대서양, 남서대서양, 남서대서양, 남서대서양, 남서대서양, 남서대

A4.
### 응답:
  한국음악은 한국에서 시작된 음악입니다. 한국음악은 전통 음악과 현대 음악으로 나눌 수 있습니다. 전통 음악은 한국의 역사와 문화를 반영하는 음악입니다. 한국의 전통 음악은 주로 가요, 단요, 지장, 보가요 등으로 구성됩니다. 현대 음악은 한국의 현대 음악을 구성하는 음악입니다. 한국의 현대 음악은 주로 pop, rock, hip-hop, R&B, electronic, jazz 등으로 구성됩니다. 한국음악은 한국의 음악의 다양한 장르로 나눌 수 있으며, 한국의 음악

```

### 3.1-8B
```
A1.
### Response:
The Eiffel Tower is a famous tall tower in Paris. It was built for the 1889 World's Fair. The tower was the tallest structure in the world at the time, standing at 324 meters. It has become an iconic symbol of Paris and one of the most recognizable landmarks globally. Visitors can ascend to the top for panoramic views of the city. The Eiffel Tower is a must-visit attraction for anyone traveling to Paris.

(Note: The response is written in a neutral, informative tone, suitable for a general audience.) <|eot_id|>

A2.
### 응답:
  ```python
def 피보나치(n):
if n <= 0:
    return "피보나치 수열은 0 이상의 정수만 지원합니다."
elif n == 1:
    return 1
elif n == 2:
    return 1
else:

A3.
### 응답:
  지구는 태양계에서 가장 큰 천체입니다. 지구는 달과 함께 태양을 공전하고 있습니다. 지구는 대기와 물로 구성된 생명체가 살아가는 세계입니다. 지구는 지구력과 지구질을 가지고 있으며 지구력은 지구가 지구질을 가지고 있는지에 대한 질문입니다. 지구질은 지구의 물질을 지구력으로부터 분리하는 것입니다. 지구력은 지구의 물질을 지구질로부터 분리하는 것입니다. 지구는 지구력과 지구질

A4.
### 응답:
  - 한국음악은 다양한 장르와 스타일로 유명합니다.
  - 트로트, 발레, 클래식, 뮤지컬, 팝, 힙합 등이 있습니다.
  - 한국음악은 다양한 문화와 역사적 유산을 반영합니다.
  - 한국음악은 세계적으로도 인기가 있습니다. - 한국음악은 다양한 장르와 스타일로 유명합니다.
  - 트로트, 발레, 클래식, 뮤지컬, 팝, 힙합 등이 있습니다.
  - 한국
```

The English responses are not bad, but the models' Korean ability and common sense are poor, especially 1B model cannot understands and responds in Korean. 

Here are the model repos: [lora](https://huggingface.co/disin7c9/LoRA_Llama-3.1-8B-Instruct-bnb-4bit_alpaca-gpt4-ko) and [gguf](https://huggingface.co/disin7c9/LoRA_Llama-3.1-8B-Instruct-bnb-4bit_alpaca-gpt4-ko-GGUF).


# 4. Conclusion

Well, it was predictable result because not only CPT is skipped but also they are lightweight models.

Also considering that many institutes are training more huge models with better architecture, data, loss functions and techniques at this very moment,
it is definitely too greedy that to build a perfect model with only this resources I just used.

However the notable points from this practice are:

1. **Now we know how to fine-tune LLMs.**
2. **We understood some important mechanisms and options in fine-tuning.**
3. **We can fine-tune LLMs more faster and memory-efficiently.**


# 5. Other informative posts

The followings are blog posts what I had seen during studying.
They are categorized into several types.

### Unsloth

#### text

https://huggingface.co/blog/mlabonne/sft-llama3 (ENG)

https://devocean.sk.com/blog/techBoardDetail.do?ID=166285&boardType=techBlog (KOR)

https://unfinishedgod.netlify.app/2024/06/15/llm-unsloth-gguf/ (KOR)

#### multimodal

https://blog.futuresmart.ai/fine-tune-llama-32-vision-language-model-on-custom-datasets (ENG)

### PEFT and TRL only

#### English data

https://kickitlikeshika.github.io/2024/07/24/how-to-fine-tune-llama-3-models-with-LoRA.html (ENG)

#### Korean data

https://naakjii.tistory.com/138 (KOR)


# References

[^1]: Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

[^2]: Dettmers, Tim, et al. "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale." Advances in Neural Information Processing Systems 35 (2022): 30318-30332.

[^3]: Ma, Shuming, et al. "The era of 1-bit llms: All large language models are in 1.58 bits." arXiv preprint arXiv:2402.17764 (2024).

[^4]: Wang, Jinheng, et al. "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1. 58 Inference on CPUs." arXiv preprint arXiv:2410.16144 (2024).

[^5]: https://unsloth.ai/blog/gradient (2024-11-21)

[^6]: Kalajdzievski, Damjan. "A rank stabilization scaling factor for fine-tuning with lora." arXiv preprint arXiv:2312.03732 (2023).

[^7]: Shuttleworth, Reece, et al. "LoRA vs Full Fine-tuning: An Illusion of Equivalence." arXiv preprint arXiv:2410.21228 (2024).
