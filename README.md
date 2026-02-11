# Semantle Helper (Word2Vec)

Game URL: https://semantle.com/

A Semantle-focused helper that uses one approach only: **word2vec vectors via `gensim`**.

This matches Semantle’s core idea (semantic distance in word2vec space) better than mixing multiple embedding systems.

## Why This Approach

Semantle states that it is based on word2vec proximity. This helper therefore uses:
- pretrained word2vec-style vectors
- cosine similarity between vectors
- rank-driven clue solving (`WORD:RANK`)

No alternate backend is used in this project anymore.

## Features

- **Neighbor mode**: for a guessed word, return top semantic neighbors.
- **Clue mode**: combine multiple Semantle clues (`word:rank`) and rank likely target words.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `gensim` for pretrained vectors
- `numpy` for vector math
- `wordfreq` for candidate-word pool construction

## First-Run Model Download

Default model is:
- `word2vec-google-news-300`

This is large and downloaded automatically by `gensim` on first use.

If you want smaller download size, you can use:
- `glove-wiki-gigaword-300`

Example:

```bash
python3 semantle_helper.py --gensim-model glove-wiki-gigaword-300 --word ocean
```

## Usage

### 1) Neighbor mode

```bash
python3 semantle_helper.py --word ocean --topk 10
```

### 2) Clue mode (Semantle feedback)

```bash
python3 semantle_helper.py \
  --clue ocean:850 \
  --clue river:320 \
  --clue water:45 \
  --topk 10
```

## CLI Options

- `--word`: one word for neighbor suggestions
- `--clue WORD:RANK`: Semantle proximity clue (repeat flag for multiple clues)
- `--topk`: number of results to print (default `10`)
- `--gensim-model`: vector model name (default `word2vec-google-news-300`)
- `--candidate-size`: candidate pool size for clue mode (default `50000`)

Notes:
- Use either `--word` or `--clue` in a single run.
- If no flags are given, interactive neighbor mode starts.

## How Clue Mode Works

Given clues like `water:45`, the helper does the following:

1. Builds a candidate word pool from frequent English words (`wordfreq`) that also exist in the selected vector model.
2. For each clue word, computes similarity from that clue to every candidate.
3. Converts the observed Semantle rank into a target similarity band.
4. Scores each candidate by weighted distance to those target similarity bands.
   - Lower rank clues (hotter clues) get higher weight.
5. Returns candidates with the best combined score.

Output columns:
- `score`: inverse-loss style score (higher is better)
- `avg_rank_error`: average absolute rank mismatch across all clues (lower is better)

## Caching and Performance

Clue mode creates a local cache file:
- `semantle_candidates_<model>_<candidate_size>.npz`

This stores candidate words and normalized vectors, so repeated clue solving is much faster.

Performance tips:
- Keep `--candidate-size` at `50000` for quality/speed balance.
- Increase to `80000+` for more coverage, decrease for speed.
- Add more high-quality clues to reduce noise in results.

## Limitations

- This is still an approximation, not Semantle’s exact internal model/corpus.
- Some clue words may be out-of-vocabulary depending on model.
- Proper nouns and morphology can behave differently across models.
