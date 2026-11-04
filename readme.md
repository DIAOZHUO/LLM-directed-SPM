

## Installation

Before installing the dependencies, please configure [PyTorch](https://pytorch.org/) according to your specific CUDA version, 
and follow the [Unsloth installation instructions](https://unsloth.ai/docs/get-started/install) to setup the Unsloth environment. 
Once these steps are complete, run the following command to install the remaining packages.

```
pip install gradio pymupdf4llm openai matplotlib deepeval websockets spmutil
```

The evaluation scripts added in the revision (prompt-enhanced baselines, GPU energy measurement, figures) additionally require:
```
pip install scikit-learn seaborn nvidia-ml-py
```
The prompt-enhanced baselines are served with [LM Studio](https://lmstudio.ai/) (`lms` CLI), 
and the GEval scores of the knowledge-base SLM are computed with Claude Sonnet 4.6 through the [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI.

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
The knowledge-base dataset used in the manuscript is not included, because it is derived from copyrighted textbooks and papers.





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


## Evaluation

The scripts in [spm_gpt/eval](spm_gpt/eval) are run from that folder, with the repository root on `PYTHONPATH`.

| Script | Content | Results |
|--------|---------|---------|
| [generate_eval_router_dataset_answer.py](spm_gpt/eval/generate_eval_router_dataset_answer.py) | router SLM | `eval_results_router` |
| [generate_eval_dataset_answer.py](spm_gpt/eval/generate_eval_dataset_answer.py), [add_claude_geval.py](spm_gpt/eval/add_claude_geval.py) | knowledge-base SLM; GEval with Claude Sonnet 4.6 as the judge (only the scores are provided, see above) | `eval_results` |
| [generate_eval_cmd_dataset_answer.py](spm_gpt/eval/generate_eval_cmd_dataset_answer.py) | command SLM, Stage i: base, quantized and fine-tuned models | `eval_results_cmd` |
| [generate_eval_planning_dataset_answer.py](spm_gpt/eval/generate_eval_planning_dataset_answer.py) | command SLM, Stage ii (67 queries) | `eval_results_cmd_planningv2` |
| [generate_eval_cmd_dataset_prompt_enhanced.py](spm_gpt/eval/generate_eval_cmd_dataset_prompt_enhanced.py) | prompt-enhanced baselines (unfine-tuned Phi-4 and Qwen3.8-27B; prompt, function calling and structured output; k = 0–20 few-shot examples) and the fine-tuned Phi-4 converted to GGUF, served with LM Studio, with the GPU energy and latency of each query | `eval_results_cmd`, `eval_results_cmd_planningv2` |

- **Exact-match criterion.** A command sequence is correct if it contains the same commands with the same arguments as the reference, in the same order; only consecutive scan-parameter settings (e.g. scan size in X and Y) and consecutive `TipFix`/`SwitchScanarea` may be reordered ([cmd_eval_util.py](spm_gpt/eval/cmd_eval_util.py)). [update_cmd_eval_accuracy.py](spm_gpt/eval/update_cmd_eval_accuracy.py) rescores existing results with this criterion and also stores the accuracy with the order ignored (`accuracy_unordered`, criterion of the first version).
- **Stage ii results.** `eval_results_cmd_planningv2` contains the results on the 67-query Stage ii test set. `eval_results_cmd_planning` contains the results on the 34 original queries, including OpenAI o4-mini (`gpt4.json`), which was no longer available through the OpenAI API when the test set was extended; `eval_results_cmd_planningv2/gpt4.json` is a copy of this 34-query result.
- **Energy.** [gpu_energy_meter.py](spm_gpt/eval/gpu_energy_meter.py) reads the cumulative energy counter of the GPU (NVIDIA Management Library); the idle power measured with the model loaded is subtracted (`net_energy_per_sample_j`). For the comparison under the same serving conditions, the fine-tuned Phi-4 was merged into the 16-bit base model with [merge_lora_to_16bit.py](spm_gpt/finetune/merge_lora_to_16bit.py) and converted to GGUF (Q4_K_M) with llama.cpp.
- **Few-shot examples.** The k examples are the training samples most similar to the query (TF-IDF character n-grams); their indices in `spmcmd_train_dataset` are recorded in `raw_outputs` of each result file.


## A part framework of our SPM instrument 


In [LocalLLM](instrument_example/LocalLLM) folder, we provide a reference implementation illustrating the communication with a locally deployed language model server, together with the associated text-parsing and command-execution modules.

The language-model components (the router, knowledge-base, and command SLMs) and the text parser interact with the instrument only through the command API ([spm_command.csv](instrument_example/LocalLLM/spm_command.csv), Table S1). 
What is tied to our hardware is the implementation of these API functions, which relies on our home-built control system and proprietary low-level drivers; 
this is why a fully executable version of the control software cannot be released, and the files that call the instrument framework 
([text2command_parser.py](instrument_example/LocalLLM/text2command_parser.py), [LocalLLMModule.py](instrument_example/LocalLLM/LocalLLMModule.py)) are intended for reference only. 
Porting the framework to another SPM requires implementing the same API functions on the target controller and adapting the fine-tuning dataset to the specifications of the target instrument (e.g., its scan range limits), while the rest of the architecture can be reused.

### Text parser

The command sequence generated by the command SLM is validated as a whole before any command is executed ([command_validator.py](instrument_example/LocalLLM/command_validator.py)): 
every line must match a command of the API with an argument of the correct type, the arguments must lie within the numerical bounds of the instrument, 
and each command is checked against the instrument state resulting from the preceding commands (e.g. a relative displacement is checked against the probe position reached by the preceding commands, and a scan is accepted only if the scan area lies within the accessible range). 
If any command is invalid, the entire sequence is rejected and no command is executed. 
During execution, each command is checked again against the current instrument state, and the remaining commands are discarded if a module reports a failure through its completion event.











