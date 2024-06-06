---
language: []
library_name: sentence-transformers
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dataset_size:1K<n<10K
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
widget:
- source_sentence: อยากกินของหวานที่มีรสส้ม
  sentences:
  - อยากกินของหวานที่มีกลิ่นส้ม
  - อยากได้ของหวานที่มีมะนาวที่มีรสชาติหวาน
  - แนะนำเมนูของหวานที่ใส่ผลไม้และน้ำผึ้งด้วยหน่อย
- source_sentence: อยากกินพายที่ใส่ผลไม้สด
  sentences:
  - อยากทานของหวานที่เป็นพายที่มีเนื้อผลไม้
  - อยากกินของหวานที่มีทั้งกาแฟและช็อคโกแลต
  - แนะนำเมนูมูสช็อกโกแลตที่ไม่หวานมาก
- source_sentence: อยากกินของหวานที่มีแครอท
  sentences:
  - ของหวานที่มีแครอทและรสหวานอมเปรี้ยวหน่อย
  - อยากกินของหวานที่มีรสชาติมะม่วงและมะนาว
  - อยากกินของหวานที่มีรสชาติคล้ายผลไม้
- source_sentence: อยากกินขนมอบไส้แอปเปิล
  sentences:
  - เมนูของหวาน อยากกินขนมอบไส้ผลไม้หวานๆ
  - อยากกินของหวานสตรอว์เบอร์รี่ไม่หวานมาก
  - อยากกินของหวานที่มีรสชาติหวานละมุนจากถั่ว
- source_sentence: อยากกินของหวานที่มีถั่ว
  sentences:
  - อยากทานของหวานที่มีส่วนผสมของถั่ว
  - อยากกินพายหวานที่ผสมเครื่องเทศอบอุ่นใจ
  - มีเมนูของหวานที่เป็นเชอร์เบทรสอะไรก็ได้มั้ย
pipeline_tag: sentence-similarity
---

# SentenceTransformer based on sentence-transformers/paraphrase-multilingual-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) <!-- at revision 79f2382ceacceacdf38563d7c5d16b9ff8d725d6 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: XLMRobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'อยากกินของหวานที่มีถั่ว',
    'อยากทานของหวานที่มีส่วนผสมของถั่ว',
    'อยากกินพายหวานที่ผสมเครื่องเทศอบอุ่นใจ',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 3,175 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | float                                                         |
  | details | <ul><li>min: 2 tokens</li><li>mean: 15.46 tokens</li><li>max: 35 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 30.21 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                           | sentence_1                                                                                                        | label            |
  |:---------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>อยากกินขนมที่ทำจากช็อกโกแลตและรสชาติเหมือนกาแฟ</code>          | <code>เมนูของหวาน เมนู Mocha มีวัตถุดิบช็อกโกแลต</code>                                                           | <code>1.0</code> |
  | <code>อยากกินของหวานที่มีรสพีชและมีถั่วด้วย</code>                   | <code>เมนูของหวาน เมนู Peach Praline Semifreddo with Amaretti มีวัตถุดิบอัลมอนด์ อัลมอนด์ พีช อัลมอนด์ พีช</code> | <code>1.0</code> |
  | <code>มีเมนูของหวานที่หอมละมุนทั้งกลิ่นผลไม้และเครื่องเทศมั้ย</code> | <code>เมนูของหวาน เมนู Peach-Cherry Lambic Charlotte มีวัตถุดิบเชอร์รีเชอร์รีน้ำผึ้งพีชมะนาวมะนาว</code>          | <code>1.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 20
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 20
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch   | Step | Training Loss |
|:-------:|:----:|:-------------:|
| 2.5126  | 500  | 1.2242        |
| 5.0251  | 1000 | 0.4793        |
| 7.5377  | 1500 | 0.1693        |
| 10.0503 | 2000 | 0.0658        |
| 12.5628 | 2500 | 0.025         |
| 15.0754 | 3000 | 0.0107        |
| 17.5879 | 3500 | 0.0051        |


### Framework Versions
- Python: 3.10.13
- Sentence Transformers: 3.0.0
- Transformers: 4.39.3
- PyTorch: 2.1.2
- Accelerate: 0.29.3
- Datasets: 2.18.0
- Tokenizers: 0.15.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply}, 
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->