# DataCleaner-llm

Let LLM do heavy corpus cleaning task!

## Main Work Flow

### for pretraining corpus

send files to llm, use it clean corpus!

1. preprocess(regex/fold unprintables)
2. docs to llm
3. llm send back cleaned texts
4. save it

### for sft datasets

1. modified "self-distillation", ref: https://arxiv.org/abs/2402.13669
2. save.

## todos

+ [x] find a proper prompt
+ [x] text chunk generator
+ [x] save the docs
+ [x] write a prompt similar to self-distillation
+ [ ] refactoring the code