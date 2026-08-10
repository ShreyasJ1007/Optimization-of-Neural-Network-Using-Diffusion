# How the toy task works

A plain explanation of what this code does, for anyone reading the repo cold.

---

## The problem in one picture

Imagine three clouds of dots scattered on a page — one red, one blue, one green.
Each cloud has a **centre** and a **spread** (how tightly the dots bunch around
that centre).

Describing all three clouds takes **12 numbers**:

```
[ centre₁ₓ, centre₁ᵧ, spread₁ₓ, spread₁ᵧ,      ← cloud 1
  centre₂ₓ, centre₂ᵧ, spread₂ₓ, spread₂ᵧ,      ← cloud 2
  centre₃ₓ, centre₃ᵧ, spread₃ₓ, spread₃ᵧ ]     ← cloud 3
```

Those 12 numbers **are the classifier**. Given a new dot, you ask which cloud
most likely produced it, and that is your prediction.

Normally you would find those 12 numbers with a formula: average the red dots to
get the red centre, and so on. **This project trains a diffusion model to
generate them instead** — with no gradient descent at generation time.

---

## Why this task

Because the correct answer is available **in closed form**. That gives two things
nothing else would:

- **Free training data.** 8,192 solved problems generate in about a second. No
  SGD needed to produce them.
- **An exact ceiling.** Every generated answer can be scored against the
  provably-best one, so "0.922" means something rather than floating in space.

---

## What goes in, what comes out

**Input — the mini-batch.** A pile of labelled dots, shape `(N, 5)`:

```
[  -0.165,  -5.082,   0, 0, 1  ]     ← position, then one-hot label
   └─ where it is ─┘  └ which cloud ┘
```

Width is always 5: two coordinates plus a three-slot class label. Height varies —
64 dots per cloud gives 192 rows. The dots are fed **raw**; handing over
pre-computed cloud centres would be handing over the answer.

**Output — 12 numbers.** The centres and spreads. That is the trained classifier.

So the model compresses ~960 numbers describing the data into the 12 that
describe the solution.

---

## The algorithm

Diffusion works by learning to **remove noise**. Training teaches it to undo
corruption; generation starts from pure noise and undoes all of it.

### Training (Algorithm 1)

Repeat thousands of times:

1. **Take a solved problem** — a correct set of 12 numbers from the pool.
2. **Pick a corruption level** — anywhere from barely touched to total noise.
3. **Corrupt it** — add random noise at that level, and *remember exactly what
   was added*.
4. **Ask the model to identify the noise** — it sees the corrupted 12 numbers,
   the noise level, and a summary of the dots, and must guess what was added.
   It never computes a gradient of the classification loss.
5. **Correct it** — compare the guess to the truth, update the network.

The dots matter here: "what good parameters look like" depends on where the
clouds actually are, so the model is told which problem it is solving.

### Generation (Algorithm 2)

For a problem the model has **never seen**:

1. **Start from 12 random numbers** — pure noise.
2. **Denoise in 50 small steps.** At each step the model looks at the current
   numbers and the dots, says which direction to nudge, and takes a small step.
3. **Return the result** — that is the trained classifier.

No gradient descent anywhere in this half.

---

## Where gradient descent does appear

Only in Stage 1, to train the optimiser — never to find the parameters it
generates.

| Stage | Gradient descent? |
|---|---|
| 1. Train the diffusion model | **Yes** |
| 2. Generate parameters | **No** — 50 forward passes |
| 3. Test by classifying held-out dots | n/a |

This is the same arrangement as any generative model: training an image
diffusion model uses gradient descent, but generating an image does not.

---

## The files

| File | Plain description |
|---|---|
| `toytask.py` | Invents dot-cloud problems, knows their correct answers, scores any guess |
| `gradtts1d.py` | The engine — a network that looks at noisy numbers and guesses the noise |
| `context.py` | Reads the pile of dots and squeezes it into a short summary |
| `paramtokens.py` | Labels which numbers mean what, so "3 clouds in 2D" and "2 clouds in 3D" aren't confused |
| `alg1.py` | The trainer and the generator — the two algorithms above |
| `tokenopt.py` | An alternative engine designed to handle any number of clouds (does not yet work) |


---

## What the results mean

**0.922 vs 0.939.** Generated parameters classify almost as well as the
mathematically perfect ones, on problems never seen during training.

**The tracking result.** As the mini-batch shrinks the *optimum itself* gets
worse — four dots is a poor estimate of a cloud's centre. The model gets worse by
roughly the same amount, following the ceiling down. A model reproducing
memorised parameter values would have stayed flat instead. So it is genuinely
computing statistics from whatever data it is given.

**Conditioning placement matters enormously.** Feeding the dot-summary in once at
the start closes 4.3% of the gap; feeding it into every layer closes 42.8%. Same
model, different plumbing.

---

## Honest limitations

- The task is **easy by construction** — mapping dots to centres and spreads is a
  smooth, simple function. This shows the machinery works end to end, not that it
  handles hard optimisation.
- Training targets came from a **formula**. For a real network no formula exists,
  and targets would have to be produced by gradient descent — the very cost the
  method aims to avoid. It only pays off if one trained model then generates
  **many** good parameter sets, spreading that cost. That is the project's central
  open question.
