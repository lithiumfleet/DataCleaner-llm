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

nop

## todos

+ [ ] find a proper prompt
+ [ ] save the docs