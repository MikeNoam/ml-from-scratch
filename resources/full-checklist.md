# ML/AI Deep Understanding — From Scratch to Core Mastery

**Goal:** Build a small language model from raw tensors, understanding every component. No API wrappers. No magic.

**Litmus Test:** Given a corpus of Mishnaic Hebrew, build a transformer from scratch in PyTorch that predicts the next token — and explain *why* each piece exists.

**Pacing:** ~10–12 hrs/week (adjust to your TaryagAI sprint load)
**Total Duration:** ~18 weeks (~4.5 months)

---

## Phase 1: Bare Metal Foundations (Weeks 1–5)
*Karpathy's "Neural Networks: Zero to Hero"*
*Analogy: Learning the shorashim before you can parse a pasuk*

### Week 1 — Micrograd: Backprop from Nothing
- [ ] Watch: "The spelled-out intro to neural networks and backpropagation" (Karpathy Lecture 1)
- [ ] Code along: Build the entire `micrograd` engine yourself — `Value` class, forward pass, backward pass
- [ ] Concept checkpoint: Draw the computation graph for `(a * b + c).backward()` by hand on paper
- [ ] Explore: Add a new operation (e.g., division, power) to your micrograd and verify gradients numerically
- [ ] **Journal prompt:** "What is a gradient, in my own words? Why does the chain rule matter?"

### Week 2 — Makemore Part 1: Bigram Language Model
- [ ] Watch: "The spelled-out intro to language modeling" (Karpathy Lecture 2)
- [ ] Code along: Build a bigram character-level model from scratch
- [ ] Concept checkpoint: What is a likelihood? What does "maximize log-likelihood" mean intuitively?
- [ ] Explore: Train it on a Hebrew names dataset (pull from Sefaria or build your own list)
- [ ] **Journal prompt:** "How is a bigram model like a Markov chain? What can't it capture?"

### Week 3 — Makemore Part 2: MLP (Multi-Layer Perceptron)
- [ ] Watch: "Building makemore Part 2: MLP" (Karpathy Lecture 3)
- [ ] Code along: Implement Bengio et al.'s neural language model from the 2003 paper
- [ ] Concept checkpoint: What is an embedding? Why do we embed characters into a continuous space?
- [ ] Key idea: Embedding is like mapping discrete letters to coordinates in a semantic space — a gematria that the network learns for itself
- [ ] Explore: Visualize your learned embeddings in 2D (PCA or t-SNE). Do similar characters cluster?
- [ ] **Journal prompt:** "What changes when we go from a lookup table (bigram) to a learned representation (MLP)?"

### Week 4 — Makemore Parts 3 & 4: Activations, BatchNorm, Becoming a Backprop Ninja
- [ ] Watch: "Building makemore Part 3: Activations & Gradients" (Karpathy Lecture 4)
- [ ] Watch: "Building makemore Part 4: Becoming a Backprop Ninja" (Karpathy Lecture 5)
- [ ] Code along: Implement BatchNorm from scratch. Manually derive every gradient in the backward pass
- [ ] Concept checkpoint: Why do activations "die"? What does BatchNorm fix?
- [ ] Key idea: BatchNorm is like normalizing vowel lengths across dialects so the model can compare apples to apples
- [ ] **Hard exercise:** Derive the backward pass for a single linear layer + tanh by hand on paper. Verify with PyTorch autograd
- [ ] **Journal prompt:** "What breaks when gradients vanish or explode? Why does initialization matter?"

### Week 5 — Makemore Part 5 + nanoGPT: Building a GPT from Scratch
- [ ] Watch: "Let's build GPT: from scratch, in code, spelled out" (Karpathy Lecture 7 — this is the crown jewel)
- [ ] Code along: Build the **entire GPT architecture** from scratch — token embeddings, positional embeddings, self-attention, multi-head attention, feedforward blocks, layer norm, the training loop
- [ ] Concept checkpoint: What is self-attention doing? Why "queries, keys, and values"?
- [ ] Key idea: Self-attention is like building a concordance on the fly — every word gets to look at every other word and decide how much to attend to it, the way a talmid chacham mentally cross-references while reading
- [ ] Train your nanoGPT on Shakespeare (as in the video), then on any text you want
- [ ] **Journal prompt:** "What are the minimal components of a transformer? Could I name and explain each one to a friend?"

**Phase 1 Milestone:** You can build a GPT from raw PyTorch tensors and explain every line.

---

## Phase 2: Classical ML Toolkit (Weeks 6–10)
*Géron "Hands-On ML", Part 1 (Chapters 1–8)*
*Analogy: Learning the Rishonim before the Acharonim — these aren't obsolete, they're the conceptual vocabulary*

### Week 6 — The ML Landscape + End-to-End Project (Ch. 1–2)
- [ ] Read: Chapters 1–2
- [ ] Code along: The California housing dataset end-to-end project
- [ ] Concept checkpoint: What is the difference between supervised, unsupervised, and reinforcement learning?
- [ ] Concept checkpoint: What is a validation set and why is it different from a test set?
- [ ] Key idea: The train/val/test split is like having a rav hamachshir who never saw the product during production — his kashrut check only means something if he's independent
- [ ] **Hands-on:** Take any CSV dataset you care about and run the full pipeline: load, explore, clean, split, train, evaluate
- [ ] **Journal prompt:** "What are the steps of an ML project, and what can go wrong at each one?"

### Week 7 — Classification + Training Models (Ch. 3–4)
- [ ] Read: Chapter 3 (Classification) — precision, recall, F1, ROC curves, confusion matrices
- [ ] Read: Chapter 4 (Training Models) — linear regression closed-form, gradient descent, polynomial regression, regularization
- [ ] Code along: All notebook exercises for both chapters
- [ ] Concept checkpoint: What is the bias-variance tradeoff?
- [ ] Key idea: Bias-variance is like pshat vs. drash — too much bias (pshat only) and you miss real patterns; too much variance (drash on everything) and you're fitting noise, seeing meaning where there is none
- [ ] **Hands-on:** Implement gradient descent from scratch (no sklearn). Plot the loss curve. Feel it converge.
- [ ] **Journal prompt:** "What is regularization protecting me from? L1 vs L2 — when would I choose each?"

### Week 8 — SVMs + Decision Trees (Ch. 5–6)
- [ ] Read: Chapter 5 (SVMs) — linear, nonlinear, kernel trick
- [ ] Read: Chapter 6 (Decision Trees)
- [ ] Code along: Notebook exercises
- [ ] Concept checkpoint: What is the "kernel trick" and why is it powerful?
- [ ] Key idea: The kernel trick is like the Rambam's negative theology — you can't describe the boundary directly in this space, so you implicitly map to a higher space where it becomes simple, without ever actually going there
- [ ] **Hands-on:** Train an SVM and a Decision Tree on the same dataset. Compare. When does each win?
- [ ] **Journal prompt:** "When would I choose a simple model over a complex one, even if the complex one scores higher?"

### Week 9 — Ensemble Methods + Dimensionality Reduction (Ch. 7–8)
- [ ] Read: Chapter 7 (Ensemble Learning) — bagging, boosting, Random Forests, XGBoost
- [ ] Read: Chapter 8 (Dimensionality Reduction) — PCA, t-SNE
- [ ] Code along: Notebook exercises
- [ ] Concept checkpoint: Why does combining weak learners produce a strong learner?
- [ ] Key idea: Ensemble learning is like a Sanhedrin — no single judge is infallible, but the majority vote of many informed judges converges toward truth (Condorcet's jury theorem in ML form)
- [ ] **Hands-on:** Build a Random Forest, inspect feature importances, visualize a few individual trees
- [ ] **Journal prompt:** "What is the 'curse of dimensionality' and why does it matter for Hebrew NLP where feature spaces can be huge?"

### Week 10 — Phase 2 Consolidation + Mini-Project
- [ ] Review all journal entries from Weeks 6–9
- [ ] **Mini-project:** Pick a small structured dataset relevant to your interests and do a full ML pipeline:
  - [ ] Data exploration and cleaning
  - [ ] Try 3+ different model types (linear, tree-based, SVM)
  - [ ] Proper cross-validation
  - [ ] Hyperparameter tuning
  - [ ] Final evaluation and write-up
- [ ] **Self-test:** Without looking at notes, explain gradient descent, regularization, bias-variance, and ensemble methods to an imaginary study partner

**Phase 2 Milestone:** You have the classical ML vocabulary. "Overfitting," "regularization," "cross-validation," "feature engineering" are now tools you can wield, not buzzwords.

---

## Phase 3: Deep Learning with Real Tensors (Weeks 11–16)
*Géron Part 2 (Chapters 10–16) + PyTorch*
*Analogy: Building from Chumash + Mishnah into Gemara — the real complexity, with the foundations to handle it*

### Week 11 — Neural Networks with Keras/PyTorch (Ch. 10)
- [ ] Read: Chapter 10 (Intro to ANNs with Keras)
- [ ] **Translate to PyTorch:** Rewrite every Keras example in PyTorch. This is where your real learning happens.
- [ ] Concept checkpoint: What is a "computational graph" in PyTorch? How does `autograd` relate to your micrograd from Week 1?
- [ ] **Hands-on:** Build a feedforward network in PyTorch for MNIST digit classification. Get >97% accuracy.
- [ ] **Journal prompt:** "How does PyTorch's autograd compare to the micrograd I built? What did they abstract away?"

### Week 12 — Training Deep Neural Networks (Ch. 11)
- [ ] Read: Chapter 11 — vanishing/exploding gradients, transfer learning, optimizers (Adam, SGD+momentum), learning rate scheduling, regularization techniques (dropout, early stopping)
- [ ] Code along, translating to PyTorch
- [ ] Concept checkpoint: Why does Adam work better than vanilla SGD in practice?
- [ ] Key idea: Adam is like a talmid who adjusts his learning pace per-topic — spending more time on hard sugyot (large gradients, slow convergence) and breezing through familiar ones
- [ ] **Hands-on:** Train the same network with SGD, SGD+momentum, and Adam. Plot all three loss curves on one chart. See the difference.
- [ ] **Journal prompt:** "What is dropout *really* doing? How is it related to ensembles?"

### Week 13 — CNNs: Learning to See (Ch. 14)
- [ ] Read: Chapter 14 (Convolutional Neural Networks)
- [ ] Code along in PyTorch: Build a CNN for image classification
- [ ] Concept checkpoint: What is a convolution? What are filters/kernels learning?
- [ ] Key idea: A convolutional filter sliding across an image is like scanning a text for a specific pattern (say, a שורש) — the filter doesn't care *where* the pattern appears, just *whether* it appears. Translation invariance.
- [ ] **Hands-on:** Visualize learned filters from your trained CNN. What patterns did it discover?
- [ ] Optional but fun: Try classifying Hebrew letter images (print vs. script ktav)
- [ ] **Journal prompt:** "Why is parameter sharing (convolution) more efficient than fully connected layers for spatial data?"

### Week 14 — RNNs and Sequence Models (Ch. 15)
- [ ] Read: Chapter 15 (RNNs, LSTMs, GRUs)
- [ ] Code along in PyTorch: Build an LSTM for text generation
- [ ] Concept checkpoint: What is the "vanishing gradient problem" in RNNs? How do LSTMs solve it?
- [ ] Key idea: An LSTM's "cell state" is like a masorah — a separate channel of information that flows forward through time, protected from corruption, carrying long-range context that the moment-to-moment processing might otherwise lose
- [ ] **Hands-on:** Train a character-level LSTM on a Hebrew text corpus. Generate text. Marvel at the nonsense. Notice what it *does* learn (word boundaries, nikud patterns, common letter combinations).
- [ ] **Journal prompt:** "What are the limitations of RNNs that motivated the invention of transformers?"

### Week 15 — Attention and Transformers (Ch. 16 + "Attention Is All You Need")
- [ ] Read: Chapter 16 (NLP with Attention and Transformers) in Géron
- [ ] **Read the paper:** Vaswani et al., "Attention Is All You Need" (2017) — the full paper, not a summary
- [ ] Concept checkpoint: Walk through the architecture diagram. Name every component. Explain its purpose.
- [ ] Key components to understand deeply:
  - [ ] Token embeddings + positional encoding (why sinusoidal? why learned?)
  - [ ] Scaled dot-product attention (why scale by √d_k?)
  - [ ] Multi-head attention (why multiple heads?)
  - [ ] Feedforward sublayers
  - [ ] Residual connections + layer normalization
  - [ ] Encoder-decoder architecture vs. decoder-only (GPT-style)
- [ ] **Hands-on:** Go back to your nanoGPT from Week 5. Now you understand every piece. Refactor it. Add comments explaining *why* each component exists. This is the test.
- [ ] **Journal prompt:** "How is multi-head attention like having multiple commentators (Rashi, Ramban, Ibn Ezra) each attending to different aspects of the same text simultaneously?"

### Week 16 — Phase 3 Consolidation
- [ ] Review all journal entries from Weeks 11–15
- [ ] Revisit your nanoGPT — make sure you can rebuild it from memory
- [ ] **Self-test:** Whiteboard the full transformer architecture from memory. Explain each component.
- [ ] Read: Jay Alammar's "The Illustrated Transformer" (http://jalammar.github.io/illustrated-transformer/) as a visual review
- [ ] Read: Lilian Weng's "Attention? Attention!" blog post for a more mathematical treatment

**Phase 3 Milestone:** You understand deep learning architectures from neurons to transformers. You can read ML papers and follow the math.

---

## Phase 4: Capstone — Your Own Transformer on Hebrew Text (Weeks 17–18)
*Analogy: Writing your own chiddush after years of learning — synthesis, not just repetition*

### Week 17 — Data Pipeline + Training
- [ ] Choose a Hebrew text corpus (options: Mishnah from Sefaria API, Tanakh, Talmud Bavli)
- [ ] Build a character-level or subword tokenizer from scratch (BPE or similar)
  - [ ] Understand: Why is tokenization especially tricky for Hebrew? (prefixed prepositions, construct chains, nikud)
- [ ] Prepare the data pipeline: text → tokens → batches → PyTorch DataLoader
- [ ] Build your transformer model from scratch in PyTorch (no HuggingFace, no pretrained anything)
- [ ] Train it. Watch the loss curve. Adjust hyperparameters.
- [ ] **Journal prompt:** "What did I learn about Hebrew's structure from watching a model try to learn it?"

### Week 18 — Evaluation, Reflection, and Next Steps
- [ ] Generate text from your trained model. Analyze what it learned and what it didn't.
- [ ] Write a technical blog post or document explaining your model architecture, training process, and results
- [ ] **Final self-assessment:**
  - [ ] Can I explain backpropagation from first principles?
  - [ ] Can I derive gradients for a simple network by hand?
  - [ ] Can I explain the transformer architecture and *why* each component exists?
  - [ ] Can I build a training pipeline from raw data to trained model without any high-level wrappers?
  - [ ] Can I read an ML paper (e.g., BERT, GPT-2) and understand the architecture section?
- [ ] **Map your next direction:**
  - [ ] Fine-tuning and transfer learning (HuggingFace, LoRA, PEFT)
  - [ ] Hebrew-specific models (AlephBERT, DictaLM — now you'll understand what they *are*)
  - [ ] Reinforcement Learning from Human Feedback (RLHF — how Claude is trained)
  - [ ] The Kaggle DeepMind competition (you now have the foundation to engage seriously)

**Phase 4 Milestone:** You built a transformer from scratch, trained it on Hebrew text, and can explain every piece. You're no longer a user of ML. You're a practitioner.

---

## Supplementary Resources (Use as Needed, Not Required)

- **3Blue1Brown "Neural Networks" series** — extraordinary visual intuition for backprop and gradient descent. Watch any time you need a visual anchor.
- **Andrew Ng's ML Specialization (Coursera)** — great for reinforcement if a concept from Géron isn't clicking. More lecture-oriented.
- **Stanford CS231n (CNNs)** — if you want to go deeper on vision
- **Stanford CS224n (NLP)** — if you want to go deeper on NLP specifically
- **"The Little Book of Deep Learning" (François Fleuret)** — free, concise, beautiful reference
- **PyTorch official tutorials** — excellent, always up to date

---

## Weekly Rhythm (Suggested)

| Block | Hours | Activity |
|-------|-------|----------|
| Mon–Tue | 3–4 hrs | Watch/read the week's material. Take notes by hand. |
| Wed–Thu | 3–4 hrs | Code along. Build the thing. Break it. Fix it. |
| Fri | 2 hrs | Explore/experiment. Try something off-script. |
| Shabbos | — | Let it marinate. |
| Sunday | 1–2 hrs | Journal entry. Review. Consolidate. |

**Total: ~10–12 hrs/week**

---

*"אֵין הַקָּדוֹשׁ בָּרוּךְ הוּא בָּא בִּטְרוּנִיָּא עִם בְּרִיּוֹתָיו" — HaShem doesn't make unreasonable demands of His creatures (Avodah Zarah 3a). Neither should your study plan. Adjust the pace to your life. Consistency beats intensity.*
