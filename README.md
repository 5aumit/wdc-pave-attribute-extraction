# **WDC-PAVE: Product Attribute Extraction with LoRA**

Can fine-tuning a small language model improve the extraction of structured, normalized product attributes from unstructured product listings?

This project compares a pretrained **Qwen3-1.7B** model against the same model fine-tuned with **LoRA** on WDC-PAVE. The implementation is a single Google Colab notebook, covering data inspection, prompting, training, evaluation, and error analysis.

## **Task**

Given a product **title** and **description**, the model must return JSON containing:

* A product **category**, selected from the dataset's five categories.  
* The **attributes** defined for that category, with normalized values where available and `null` when information is missing.

The category is **not** supplied as input. Allowed category-specific attribute names are derived from the training split and included in the system prompt.

## **Dataset**

[WDC-PAVE attribute-value extraction dataset](https://huggingface.co/datasets/siavashsaki/wdc-pave-ave)

| Split | Products | Use |
| :---- | :---- | :---- |
| Train | 994 | Supervised fine-tuning and prompt demonstrations |
| Validation | 142 | Model comparison and error analysis |
| Test | 284 | Held-out final evaluation, **not yet run** |

The five product categories are Computers And Accessories, Home And Garden, Office Products, Jewelry, and Grocery And Gourmet Food.

## **Approach**

**Base model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

The pretrained baseline receives a category/attribute schema in its system prompt. Evaluation uses two fixed demonstrations drawn from the training split. Generation is greedy (`do_sample=False`).

For supervised fine-tuning, the model is trained to generate the ground-truth JSON from the product title, description, and schema instructions. Training prompts contain **no few-shot demonstrations**. The loss is applied only to assistant-response tokens; prompt tokens are masked with `-100`.

### **Fine-tuning configuration**

| Setting | Value |
| :---- | :---- |
| Method | LoRA (frozen base model) |
| Target modules | `q_proj`, `v_proj` |
| LoRA rank / alpha / dropout | `8` / `16` / `0.05` |
| Trainable parameters | 1,605,632 (\~0.093%) |
| Epochs | 1 |
| Learning rate | `2e-4` |
| Batch size | 1 |
| Gradient accumulation | 8 steps |
| Optimizer | AdamW (`adamw_torch`) |
| Random seed | 42 |
| Runtime used | Google Colab, NVIDIA L4 |

The notebook also contains configuration paths for SmolLM2, FLAN-T5, and other training methods; **the reported results below are for Qwen3-1.7B with LoRA only**.

## **Results**

Both models were evaluated on the **same 142 validation products**, using the **same two-shot prompts** and greedy decoding.

| Metric | Pretrained | Fine-tuned |
| :---- | :---- | :---- |
| Valid JSON | 100.0% | 99.3% |
| Category accuracy | 85.9% | **88.7%** |
| Attribute precision | 17.6% | **50.8%** |
| Attribute recall | 20.4% | **44.8%** |
| Attribute F1 | 18.9% | **47.6%** |

**Attribute F1 improved by 28.7 percentage points on validation.** Category accuracy improved more modestly. The fine-tuned model produced invalid JSON for one validation product, so formatting reliability was not perfect.

These are **validation results**, not final test results. No conclusion about held-out test performance is made here.

### **Evaluation definitions**

* **Valid JSON:** Fraction of outputs that parse into a JSON object.  
* **Category accuracy:** Fraction of products assigned their correct category.  
* **Attribute precision, recall, and F1:** Micro-averaged exact matches between predicted and ground-truth `(attribute_name, value)` pairs. Values are stripped and case-folded; attribute names must match exactly. `null` values are excluded, and list values are compared as individual pairs.

An incorrect value counts as a false-positive prediction and a missed ground-truth pair. This metric does **not** independently measure attribute presence detection or require the entire output schema to be correct.

## **Run the notebook**

Open [`Finetune-WDC-PAVE-Workbench.ipynb`](https://chatgpt.com/c/Finetune-WDC-PAVE-Workbench.ipynb) in **Google Colab** and select a GPU runtime (an L4 was used for the reported run).

1. In **Section 2**, set `DRIVE_DIR` to an existing folder in your Google Drive. The notebook writes experiment artifacts there.  
2. Start with `RUN_TRAINING = False` and `FINAL_TEST = False` to inspect the pretrained baseline. Run through the baseline evaluation; **skip the LoRA-only full-validation comparison cell** for this baseline-only run.  
3. To train a new adapter, set `RUN_TRAINING = True` and choose a **new** `EXPERIMENT_NAME`. Existing `final_model/` directories are protected from being overwritten.  
4. Rerun the notebook from the top for training. It installs its Python dependencies and downloads the dataset from Hugging Face.  
5. Compare the pretrained and fine-tuned predictions in **Section 11**. The full-validation comparison cell is specific to a loaded LoRA model with `N_SHOTS = 2`.

To reload a completed adapter instead of retraining, set `RUN_TRAINING = False` and `LOAD_CHECKPOINT_DIR` to the saved `final_model/` directory, then rerun from the start. Leave `FINAL_TEST = False` until model selection is complete.

## **Saved experiment files**

The notebook saves generated artifacts under `DRIVE_DIR / EXPERIMENT_NAME /`:

qwen3-lora-01/

├── final\_model/

│   ├── adapter\_model.safetensors    \# Trained LoRA weights

│   ├── adapter\_config.json         \# Adapter settings and base-model reference

│   └── ...                         \# Tokenizer and chat-template files

├── checkpoints/

│   └── checkpoint-125/             \# Adapter \+ optimizer/training state

├── pretrained\_full\_2shot\_validation\_predictions.jsonl

├── fine\_tuned\_full\_2shot\_validation\_predictions.jsonl

├── full\_validation\_comparison.csv

└── ...                             \# Smaller validation runs

* **For inference:** Load the Qwen3 base model and the adapter from `final_model/`. The adapter file is **not** a complete standalone 1.7B-parameter model.  
* **For resuming training:** Checkpoints additionally contain optimizer, learning-rate scheduler, random-state, and trainer-state files.  
* **For error analysis:** Prediction JSONL files store generated responses alongside the reference outputs. CSV files summarize metrics.

Model weights, checkpoints, and large generated outputs are kept outside Git. They are not required to read the notebook or review the reported results.

## **Limitations and next steps**

* Only one LoRA configuration and one training epoch have been evaluated so far.  
* The fine-tuned model was trained with zero-shot prompts but evaluated with two-shot prompts; zero-shot evaluation after fine-tuning remains a useful comparison.  
* Exact-string attribute scoring penalizes equivalent values expressed in different units or formats.  
* Attribute-presence detection and exact schema compliance are distinct from the current attribute F1 metric.  
* Performance varies across products and categories; aggregate metrics can hide regressions.  
* The 284-product test split remains untouched until model selection is complete.

## **References**

* [WDC-PAVE dataset](https://huggingface.co/datasets/siavashsaki/wdc-pave-ave)  
* [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)  
* [Hugging Face PEFT / LoRA documentation](https://huggingface.co/docs/peft/index)

&nbsp;