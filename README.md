# Semantle Helper (Word2Vec)

Game URL: https://semantle.com/

A Semantle-focused helper that uses one approach only: **word2vec vectors via `gensim`**.

## Rank Convention Used Here

This helper follows the Semantle proximity direction you requested:

- Higher rank means closer to the answer.
- Ranked guesses are in `1-999`.
- `999` is the hottest non-answer guess.
- `1` is the weakest guess that still has a displayed rank.
- If similarity is `100`, you solved the puzzle.

## Features

- **Neighbor mode**: show top semantic neighbors for a word.
- **Clue mode**: combine Semantle clues (`word:rank`) and rank likely targets.
- **Play mode**: log guesses while playing and continuously suggest next guesses.
- **Online learner (ML)**: updates after each input and blends learned target estimates with clue-based ranking.
- **Auto-status input**: `<word> <score>` works even when Semantle shows no rank.
- **Amend support**: correct typo/mistake in a recorded guess.
- **Flexible suggestions**: `suggest` defaults to 5, `suggest n` returns `n`.
- **Readable list view**: `list` prints a table/sheet view.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:

- `gensim`
- `numpy`
- `wordfreq`

## First-Run Model Download

Default model:

- `word2vec-google-news-300`

It is downloaded once by `gensim` and then cached locally.

Optional smaller model:

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

### 2) Clue mode

Provide clues as `word:rank` where rank is `1-999` and higher is hotter:

```bash
python3 semantle_helper.py \
  --clue ocean:850 \
  --clue river:320 \
  --clue water:45 \
  --topk 10
```

### 3) Play mode

```bash
python3 semantle_helper.py --play
```

You will be asked for three daily score anchors:

- nearest non-answer score (rank `999`)
- 10th-nearest non-answer score (rank `990`)
- weakest ranked score (rank `1`)

Then use commands in `play>`:

- `<word> <score>`: record/update guess (status auto-inferred)
- `<word> <score> <rank>`: record/update with explicit numeric rank (`1-999`)
- `<word>`: show top 10 similar words
- `<word> 100`: mark solved and end play mode
- `amend <old_word> <new_word> <score> [rank]`: fix typo/mistake
- `suggest`: show 5 hybrid suggestions (clue solver + online learner)
- `suggest <n>`: show `n` hybrid suggestions
- `random`: show 5 random words (default)
- `random <n>`: show `n` random words
- `list`: show recorded guesses in table format
- `remove <word>`: delete a recorded guess
- `help`: show commands
- `quit`: exit

### Play-mode anchor flags

```bash
python3 semantle_helper.py --play \
  --top999-score 52.31 \
  --top990-score 48.92 \
  --top1-score 36.80
```

Backward-compatible aliases are also accepted:

- `--top2-score` for `--top999-score`
- `--top10-score` for `--top990-score`
- `--top1000-score` for `--top1-score`

## Auto Status Rules (No Rank Shown)

When you enter `<word> <score>`:

- `far` if score `< 0`
- `cold` if score is `0-20`
- `tepid` if score `> 20`

These are converted into coarse rank constraints and combined with other clues.

## Online Learner (ML)

During play mode, each new guess is used as a training signal for a lightweight online model:

- Numeric rank clues are treated as strong supervision.
- Score-derived rank estimates are treated as medium supervision.
- Status-only clues (`far/cold/tepid`) are treated as weak supervision.

On every `suggest` call, the helper blends:

- baseline clue solver score
- online-learner similarity score

The blend weight is confidence-based, so the model relies more on online learning as more inputs are recorded.

## CLI Options

- `--word`: neighbor query mode
- `--clue WORD:RANK`: clue mode input (`1-999`, higher is closer)
- `--play`: interactive play mode
- `--suggestions`: default suggestion count in play mode (default `5`)
- `--no-online-ml`: disable online learner blending and use baseline solver only
- `--top999-score`, `--top990-score`, `--top1-score`: play-mode anchors
- `--topk`: output count in neighbor/clue mode (default `10`)
- `--gensim-model`: model name (default `word2vec-google-news-300`)
- `--candidate-size`: clue/play candidate pool size (default `50000`)

## Caching and Performance

Clue/play mode caches candidate vectors in:

- `semantle_candidates_<model>_<candidate_size>.npz`

This makes repeated runs much faster.

## Limitations

- This is an approximation, not Semantle’s exact internal corpus/model.
- OOV words may fail depending on the selected model.
- Proper nouns and inflections can behave differently than expected.
