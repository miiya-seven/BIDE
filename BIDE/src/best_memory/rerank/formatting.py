"""Explicit Qwen3 reranker input contract, independent of installed chat templates."""
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
PROTOCOL='qwen3-explicit-pair-v1'

def encode_pairs(tokenizer, query, documents, instruction=INSTRUCTION, max_length=2048):
    if not isinstance(query,str) or not query.strip():raise ValueError('query must be nonempty')
    if not documents or any(not isinstance(d,str) or not d.strip() for d in documents):
        raise ValueError('documents must contain nonempty strings')
    prefix=tokenizer.encode(PREFIX,add_special_tokens=False)
    suffix=tokenizer.encode(SUFFIX,add_special_tokens=False)
    budget=max_length-len(prefix)-len(suffix)
    if budget<=0:raise ValueError('max_length too small for rerank contract')
    texts=[f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {d}' for d in documents]
    tokens=tokenizer(texts,padding=False,truncation=True,max_length=budget,add_special_tokens=False)['input_ids']
    return [prefix+ids+suffix for ids in tokens]
