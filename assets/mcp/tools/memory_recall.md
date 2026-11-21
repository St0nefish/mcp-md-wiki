Get a short block of the memories most relevant to `query`, ready to put in front
of a model. Only `confirmed` memories are used unless `include_proposed` is set.
The block stays within `max_chars` and `max_items`; the same query over the same
memories always gives the same block.
