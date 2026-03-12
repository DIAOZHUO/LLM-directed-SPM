

## Installation

Before installing the dependencies, please configure [PyTorch](https://pytorch.org/) according to your specific CUDA version, 
and follow the [Unsloth installation instructions](https://unsloth.ai/docs/get-started/install) to setup the Unsloth environment. 
Once these steps are complete, run the following command to install the remaining packages.

```
pip install gradio pymupdf4llm openai matplotlib deepeval websockets spmutil
```

The operation of this project has been verified on NVIDIA RTX 4090 and RTX 5090 GPUs.
A minimum of 18GB VRAM is required for the full project experience.


(optional) The scripts utilize OpenAI API functionality. 
To use these features, please enter your API key in the `api_keys.openai.py` file.
```
openai_api_key = "sk-xxxxxxxxxxx"
```

## Quick Start




### Local server of the SLM-integrated System


The implementation includes three integrated Small Language Models (SLMs) and the "Dynamic LoRA Adapter Injection Scheme". 
These core features are powered by a local server, which can be launched via [main.py](ui/main.py).



### Automatic knowledge-base construction workflow


The knowledge-base construction workflow described in this manuscript is implemented through a three-stage LLM invocation process. 
To demonstrate the workflow, please execute the following scripts in order:

- [generate_dataset.ipynb](dataset/knowledge_base_construction/generate_dataset.ipynb)
- [generate_answer.ipynb](dataset/knowledge_base_construction/generate_answer.ipynb)
- [distill_dataset.ipynb](dataset/knowledge_base_construction/distill_dataset.ipynb)

The sample paper used in this demonstration (https://doi.org/10.1002/smtd.202400813) is provided under the CC BY 4.0 license.




## Fine-tuned model distributions

The models include fine-tuned and distilled versions of the Knowledge-Base SLM, alongside fine-tuned versions of the Command SLM.

All model files total 10.8 GB and need to be downloaded separately（see [how to get models](spm_gpt/finetune/how%20to%20get%20models)）


knowledge-base SLM

| Model Category      | Fine-tuned Model       | Distilled Model                |
|---------------------|------------------------|--------------------------------|
| Llama-3.2  | [Llama-3.2-3B-Instruct_unsloth_results](spm_gpt/finetune/Llama-3.2-3B-Instruct_unsloth_results) | [Llama-3.2-3B-Instruct_distill_unsloth_results](spm_gpt/finetune/Llama-3.2-3B-Instruct_distill_unsloth_results)|
| Mistral-v0.3        |  [mistral-7b-instruct-v0.3-bnb-4bit_unsloth_results](spm_gpt/finetune/mistral-7b-instruct-v0.3-bnb-4bit_unsloth_results)          | [mistral-7b-instruct-v0.3-bnb-4bit_distill_unsloth_results](spm_gpt/finetune/mistral-7b-instruct-v0.3-bnb-4bit_distill_unsloth_results) |
| Phi-4        |  [Phi-4-unsloth-bnb-4bit_unsloth_results](spm_gpt/finetune/Phi-4-unsloth-bnb-4bit_unsloth_results)| [Phi-4-unsloth-bnb-4bit_distill_unsloth_results](spm_gpt/finetune/Phi-4-unsloth-bnb-4bit_distill_unsloth_results)|


command SLM

| Model Category      | Fine-tuned Model       |
|---------------------|------------------------|
| Llama-3.2  | [LLAMA_unsloth_spmcmd](spm_gpt/finetune/LLAMA_unsloth_spmcmd) |
| Mistral-v0.3        |  [mistral_unsloth_spmcmd](spm_gpt/finetune/mistral_unsloth_spmcmd)     |
| Phi-4        |  [Phi-4_unsloth_spmcmd](spm_gpt/finetune/Phi-4_unsloth_spmcmd)|



For validation details, please refer to the [eval](spm_gpt/eval) folder. This directory contains:

- Validation Methods: The specific protocols used to benchmark model performance.

- Experimental Data: The raw data used for the evaluations presented in the manuscript.

- Visualization Code: Scripts used to generate the figures and plots found in the paper.


## A part framework of our SPM instrument 


In [LocalLLM](instrument_example/LocalLLM) folder, we provide a reference implementation illustrating the communication with a locally deployed language model server, together with the associated text-parsing and command-execution modules.
It should be noted that the SPM system used in this work is a fully custom-built instrument. The Python control framework is therefore tightly coupled to proprietary hardware interfaces and low-level drivers, 
which prevents us from releasing a fully functional and executable version of the control software. 
As a result, the provided framework is intended for conceptual and architectural reference only, rather than for direct execution.















