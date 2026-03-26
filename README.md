# ML From Scratch — From Tensors to Transformers

A hands-on learning journal: building ML/AI understanding from the ground up, culminating in a Hebrew-text transformer built from raw PyTorch tensors.

## Structure

```
ml-from-scratch/
├── phase-1-foundations/       # Karpathy Zero to Hero (Weeks 1–5)
│   ├── week-01-micrograd/
│   ├── week-02-bigram/
│   ├── week-03-mlp/
│   ├── week-04-activations-batchnorm/
│   └── week-05-nanogpt/
├── phase-2-classical-ml/      # Géron Part 1, Chapters 1–8 (Weeks 6–10)
│   ├── week-06-ml-landscape/
│   ├── week-07-classification-training/
│   ├── week-08-svm-trees/
│   ├── week-09-ensembles-dimreduction/
│   └── week-10-consolidation-project/
├── phase-3-deep-learning/     # Géron Part 2 + PyTorch (Weeks 11–16)
│   ├── week-11-pytorch-nn/
│   ├── week-12-training-deep-nets/
│   ├── week-13-cnn/
│   ├── week-14-rnn-lstm/
│   ├── week-15-attention-transformers/
│   └── week-16-consolidation/
├── phase-4-capstone/          # Hebrew transformer from scratch (Weeks 17–18)
│   ├── week-17-hebrew-transformer/
│   └── week-18-evaluation/
└── resources/                 # Papers, reference notes, useful links
```

## Each Weekly Folder Contains

| File | Purpose |
|------|---------|
| `my_*.ipynb` | My from-scratch implementation (not copied from lectures) |
| `experiments.ipynb` | "What happens if..." explorations |
| `journal.md` | Concept notes, analogies, reflections |

## Litmus Test

Can I build a small transformer from scratch in PyTorch, train it on a Mishnaic Hebrew corpus, and explain *why* every component exists?

## Tools

- **Weeks 1–10:** Google Colab (free tier)
- **Weeks 11–18:** Local on MacBook Pro 16" (M-series, PyTorch MPS backend)

## Key Resources

- [Karpathy: Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- Géron: *Hands-On Machine Learning with Scikit-Learn and PyTorch* 
- Vaswani et al.: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- [Jay Alammar: The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
