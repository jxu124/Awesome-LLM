# Awesome-LLM
    ** Awesome Large (Vision-) Language Model **
    这个项目的目的是收集一些公开发表的文章、项目、模型等大(视觉)语言模型以及一些相关的数据集。

## Projects/Models

### ❤ THUDM/GLM
- https://github.com/THUDM/GLM
- https://arxiv.org/abs/2103.10360

> GLM-10B/130B: 双语双向稠密模型。GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.

### THUDM/ChatGLM-6B
- https://github.com/THUDM/ChatGLM-6B
- https://huggingface.co/THUDM/chatglm-6b
- https://github.com/THUDM/GLM-130B
- https://arxiv.org/abs/2103.10360
- https://arxiv.org/abs/2210.02414
- https://chatglm.cn/blog

> ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考我们的博客。

### ❤ facebookresearch/metaseq（OPT）
- https://github.com/facebookresearch/metaseq
- https://huggingface.co/facebook/opt-13b
- https://arxiv.org/abs/2205.01068

> OPT-2.7B/13B/30B/66B：meta开源的大语言模型，据说效果一般，更推荐使用meta的LLaMa模型。

### ❤ facebookresearch/llama（LLaMa）
- https://github.com/facebookresearch/llama
- https://ai.facebook.com/blog/large-language-model-llama-meta-ai/
- https://arxiv.org/abs/2302.13971v1

### tloen/Alpaca-LoRA（LLaMa+Alpaca-LoRA）
- https://github.com/tloen/alpaca-lora
- https://huggingface.co/spaces/tloen/alpaca-lora

> Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. 
相关项目（这个页面里也有）：
- https://huggingface.co/decapoda-research/llama-7b-hf
- https://huggingface.co/datasets/yahma/alpaca-cleaned

### LLaMa预训练模型参数（LLaMa Pretrain）
- https://huggingface.co/decapoda-research

### LC1332/Chinese-alpaca-lora（Luotuo）
- https://github.com/LC1332/Chinese-alpaca-lora
- https://github.com/LC1332/Luotuo-Chinese-LLM

> 骆驼(Luotuo) is the Chinese pinyin(pronunciation) of camel.

### Claude
- https://arxiv.org/pdf/2212.08073.pdf

> Anthropic公司开放的接口，从OpenAI离职的团队

### BelleGroup/BELLE-7B-2M
- https://github.com/LianjiaTech/BELLE
- https://huggingface.co/BelleGroup/BELLE-7B-2M

> 中文微调的语言模型 on Bloomz-7b1-mt。BELLE is based on Bloomz-7b1-mt and finetuned with 2M Chinese data combined with 50,000 pieces of English data from the open source Stanford-Alpaca, resulting in good Chinese instruction understanding and response generation capabilities.


### ❤ mlfoundations/open_flamingo【VL】
- https://laion.ai/blog/open-flamingo/
- https://github.com/mlfoundations/open_flamingo

> 来自[LAION](https://laion.ai/about/)团队，视觉语言模型，DeepMind的[flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)的开源实现，效果略低于flamingo。

### lucidrains/flamingo-pytorch【VL】
- https://github.com/lucidrains/flamingo-pytorch （未放出预训练权重）

> 一个flamingo的（第三方？）实现，没有放出[预训练权重](https://github.com/lucidrains/flamingo-pytorch/issues/4)。Implementation of Flamingo, state-of-the-art few-shot visual question answering attention net, in Pytorch. 


### microsoft/visual-chatgpt【VL】
- https://github.com/microsoft/visual-chatgpt
- https://arxiv.org/abs/2303.04671

> 使用chatGPT连接各类基础视觉模型来实现各种任务。Visual ChatGPT connects ChatGPT and a series of Visual Foundation Models to enable sending and receiving images during chatting.

### hpcaitech/ColossalAI【VL】
- https://github.com/hpcaitech/ColossalAI
- https://arxiv.org/abs/2110.14883

> Colossal-AI 为您提供了一系列并行组件。我们的目标是让您的分布式 AI 模型像构建普通的单 GPU 模型一样简单。我们提供的友好工具可以让您在几行代码内快速开始分布式训练和推理。

### microsoft/unilm（Kosmos-1）【VL】
- https://github.com/microsoft/unilm （暂未开源）
- https://arxiv.org/pdf/2302.14045.pdf

> 微软的大型VL模型，但没有开源预训练模型。

### nomic-ai/gpt4all
- https://github.com/nomic-ai/gpt4all
- https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf

> Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa.

### OptimalScale/LMFlow
- https://github.com/OptimalScale/LMFlow
- https://lmflow.com/

> 拥有自己的 AI 大模型！开源项目 LMFlow 支持上千种模型，提供全流程高效训练方案。An extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly, speedy and reliable, and accessible to the entire community. Large Language Model for All. See our vision. 共建大模型社区，让每个人都能训得起大模型。查看我们的愿景。

### microsoft/JARVIS（HuggingGPT）
- https://github.com/microsoft/JARVIS （coming soon）
- https://arxiv.org/pdf/2303.17580.pdf

> 类似微软的visual-chatgpt，也是利用chatGPT去使用各种各样的基础模型（不过是在huggingface上的）。

### Others
<details>
<summary>点击展开</summary>

### Chat-REC
- https://arxiv.org/abs/2303.14524

> 语言模型的推荐链应用。通过将用户画像和历史交互转换为 Prompt，Chat-Rec 可以有效地学习用户的偏好，它不需要训练，而是完全依赖于上下文学习，并可以有效推理出用户和产品之间之间的联系。通过 LLM 的增强，在每次对话后都可以迭代用户偏好，更新候选推荐结果。

### BloombergGPT
- https://arxiv.org/pdf/2303.17564.pdf

> 金融领域数据的50b大语言模型。

</details>

## Datasets

### tatsu-lab/stanford_alpaca
- https://github.com/tatsu-lab/stanford_alpaca

> 目的是训练一个Instruction-following LLaMA模型，其中包括50K数据集、数据采集脚本、微调的模型。这个工作使LLaMa达到了接近chatGPT的效果，但仍存在数据量较小、不支持中文等问题。



### yahma/alpaca-cleaned
- https://huggingface.co/datasets/yahma/alpaca-cleaned
- https://github.com/gururise/AlpacaDataCleaned

> This is a cleaned version of the original Alpaca Dataset released by Stanford.

### victorsungo/MMDialog
- https://github.com/victorsungo/MMDialog

> 微软的视觉多轮对话数据集。


## Papers

    可能有用的papers们。

- [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf)



