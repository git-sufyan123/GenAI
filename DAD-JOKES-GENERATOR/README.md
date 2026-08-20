# Dad Jokes Generator (LSTM)

A small text-generation project that trains an LSTM on a hand-collected dataset of dad jokes and generates new joke-like sentences word by word.

## What it does

- Uses a corpus of dad jokes as training data
- Tokenizes the text and builds word sequences for next-word prediction
- Trains an LSTM model (Keras/TensorFlow) to predict the next word given a sequence of previous words
- Generates new text by starting from a seed phrase (e.g. `"what do"`) and repeatedly predicting the next word, feeding each prediction back in as input

This is essentially a simplified, from-scratch look at **autoregressive text generation** — the same core idea (predict next token → feed it back → repeat) that powers larger language models, just at a much smaller scale using an LSTM instead of a transformer.

## Example output

```
what do you call a group of disorganized cats a cat tastrophe
```

## Tech stack

- Python
- TensorFlow / Keras
- LSTM (word-level sequence model)
- Tokenizer + padded sequences for preprocessing

## Status

This is a learning project built to understand how sequence models and text generation work end to end — from raw text, to tokenization, to training, to generation. The model currently always picks the highest-probability next word, so outputs are deterministic for a given seed phrase. Output quality is decent but not perfect, and there's room to improve variety and coherence with further tuning.

## Possible next steps

- Add temperature/sampling-based generation instead of always picking the top prediction, for more varied outputs
- Expand the training dataset with more jokes
- Experiment with a larger/deeper LSTM or a different architecture

## Notes

Built independently as a hands-on way to learn sequence models and text generation before moving into more advanced NLP/GenAI concepts.
