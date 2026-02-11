#!/usr/bin/env python3
"""Semantle helper using a single word2vec-based approach.

Features:
- Neighbor mode: suggest words semantically close to a query.
- Clue mode: combine Semantle clues (WORD:RANK) to suggest likely targets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import gensim.downloader as gensim_api
except ImportError:  # pragma: no cover - handled at runtime
    gensim_api = None

try:
    from wordfreq import top_n_list
except ImportError:  # pragma: no cover - handled at runtime
    top_n_list = None


# -------------------- Defaults --------------------
DEFAULT_GENSIM_MODEL = "word2vec-google-news-300"
DEFAULT_CANDIDATE_SIZE = 50_000
DEFAULT_TOPK = 10


# -------------------- Data structures --------------------
@dataclass
class ClueData:
    word: str
    observed_rank: int
    sims: np.ndarray
    sorted_sims: np.ndarray
    target_sim: float
    weight: float


# -------------------- Basic helpers --------------------
def normalize_query(word: str) -> str:
    return word.strip().lower()


def is_clean_word(word: str) -> bool:
    """Keep simple alphabetic tokens that match typical Semantle guesses."""
    return word.isalpha() and 2 <= len(word) <= 24


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def cache_file_for_candidates(model_name: str, candidate_size: int) -> Path:
    return Path(f"semantle_candidates_{safe_name(model_name)}_{candidate_size}.npz")


# -------------------- Model and vocabulary loading --------------------
def load_gensim_model(model_name: str):
    """Load pretrained keyed vectors from gensim-data (download once)."""
    if gensim_api is None:
        raise RuntimeError("Missing dependency 'gensim'. Install with: pip install gensim")

    print(f"Loading gensim model '{model_name}'...")
    return gensim_api.load(model_name)


def resolve_gensim_token(query: str, model) -> str | None:
    """Try common casing variants before treating a word as out-of-vocabulary."""
    for candidate in (query, query.lower(), query.capitalize(), query.upper()):
        if candidate in model.key_to_index:
            return candidate
    return None


def unit_vector(model, token: str) -> np.ndarray:
    """Fetch and L2-normalize a token vector."""
    vec = np.asarray(model.get_vector(token), dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12)


def build_vocab_from_wordfreq(size: int, multiplier: int = 3) -> List[str]:
    """Build a frequent-word candidate vocabulary for clue solving."""
    if top_n_list is None:
        raise RuntimeError("Missing dependency 'wordfreq'. Install with: pip install wordfreq")

    raw_words = top_n_list("en", size * multiplier)
    vocab: List[str] = []
    seen = set()

    for word in raw_words:
        token = normalize_query(word)
        if not is_clean_word(token):
            continue
        if token in seen:
            continue

        seen.add(token)
        vocab.append(token)

        if len(vocab) >= size:
            break

    return vocab


# -------------------- Neighbor mode --------------------
def top_k_similar(query_word: str, k: int, model) -> List[Tuple[str, float]]:
    """Top-k nearest neighbors from word2vec vectors."""
    query = normalize_query(query_word)
    token = resolve_gensim_token(query, model)

    if token is None:
        raise ValueError(
            f"'{query_word}' is not in model vocabulary. Try another word or model."
        )

    # Pull extra neighbors so we can filter punctuation/multi-token artifacts.
    raw = model.most_similar(token, topn=max(60, k * 8))
    out: List[Tuple[str, float]] = []
    seen = {query}

    for candidate, score in raw:
        cleaned = normalize_query(candidate)
        if cleaned in seen or not is_clean_word(cleaned):
            continue

        seen.add(cleaned)
        out.append((cleaned, float(score)))
        if len(out) == k:
            break

    return out


# -------------------- Candidate matrix cache --------------------
def load_or_create_candidate_matrix(
    model,
    model_name: str,
    candidate_size: int,
) -> Tuple[List[str], np.ndarray]:
    """Build/load candidate vectors used in clue mode.

    Candidate words come from frequent English words (wordfreq) filtered to terms
    that exist in the model vocabulary.
    """
    cache_file = cache_file_for_candidates(model_name, candidate_size)

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True, mmap_mode="r")
        cached_model = data["model_name"].item()
        cached_size = int(data["candidate_size"].item())
        if cached_model == model_name and cached_size == candidate_size:
            vocab = data["vocab"].tolist()
            vectors = np.asarray(data["vectors_norm"], dtype=np.float32)
            return vocab, vectors

    seed_vocab = build_vocab_from_wordfreq(candidate_size)

    vocab: List[str] = []
    vectors: List[np.ndarray] = []

    for token in seed_vocab:
        model_token = resolve_gensim_token(token, model)
        if model_token is None:
            continue

        vocab.append(token)
        vectors.append(unit_vector(model, model_token))

        if len(vocab) >= candidate_size:
            break

    if not vocab:
        raise RuntimeError("No overlap between candidate words and model vocabulary.")

    vectors_norm = np.vstack(vectors).astype(np.float32)

    np.savez_compressed(
        cache_file,
        model_name=model_name,
        candidate_size=candidate_size,
        vocab=np.array(vocab, dtype=object),
        vectors_norm=vectors_norm,
    )

    return vocab, vectors_norm


# -------------------- Clue mode parsing --------------------
def parse_clues(clue_args: Sequence[str]) -> List[Tuple[str, int]]:
    """Parse repeated WORD:RANK clues from CLI."""
    clues: List[Tuple[str, int]] = []
    seen = set()

    for raw in clue_args:
        if ":" not in raw:
            raise ValueError(f"Invalid clue '{raw}'. Use WORD:RANK")

        word_raw, rank_raw = raw.split(":", 1)
        word = normalize_query(word_raw)

        if not is_clean_word(word):
            raise ValueError(f"Invalid clue word '{word_raw}'")

        try:
            rank = int(rank_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid rank in clue '{raw}'") from exc

        if rank <= 0:
            raise ValueError(f"Rank must be > 0 in clue '{raw}'")

        # Duplicate clue words add no information; ignore them.
        if word in seen:
            continue

        seen.add(word)
        clues.append((word, rank))

    if not clues:
        raise ValueError("No valid clues provided")

    return clues


# -------------------- Clue mode scoring math --------------------
def approx_rank_from_sorted(sorted_sims: np.ndarray, sim: float) -> int:
    """Approximate rank from similarity against sorted candidate similarities."""
    ge_count = int(len(sorted_sims) - np.searchsorted(sorted_sims, sim, side="left"))
    return max(1, ge_count - 1)


def build_clue_data(
    clues: Sequence[Tuple[str, int]],
    model,
    candidate_vectors: np.ndarray,
) -> List[ClueData]:
    """Convert clue ranks into target similarity bands for scoring."""
    clue_data: List[ClueData] = []
    n = len(candidate_vectors)

    for clue_word, observed_rank in clues:
        token = resolve_gensim_token(clue_word, model)
        if token is None:
            raise ValueError(f"Clue word '{clue_word}' is not in model vocabulary")

        clue_vec = unit_vector(model, token)
        sims = candidate_vectors @ clue_vec

        # Convert observed rank to the similarity value at that rank.
        # Rank 1 means top neighbor (excluding self).
        desc_index = min(max(observed_rank, 1), n - 1)
        kth = n - 1 - desc_index
        target_sim = float(np.partition(sims.copy(), kth)[kth])

        # Hotter clues (lower ranks) get larger influence in total loss.
        weight = 1.0 / np.log2(observed_rank + 2.0)

        clue_data.append(
            ClueData(
                word=clue_word,
                observed_rank=observed_rank,
                sims=sims,
                sorted_sims=np.sort(sims),
                target_sim=target_sim,
                weight=weight,
            )
        )

    return clue_data


def solve_from_clues(
    clues: Sequence[Tuple[str, int]],
    model,
    candidate_vocab: Sequence[str],
    candidate_vectors: np.ndarray,
    topk: int,
) -> List[Tuple[str, float, float]]:
    """Suggest likely secret words from Semantle clues.

    Returns tuples: (candidate_word, score, avg_rank_error)
    """
    clue_data = build_clue_data(clues, model, candidate_vectors)

    total_loss = np.zeros(len(candidate_vocab), dtype=np.float32)

    # Candidate score = weighted distance from each clue's implied target similarity.
    for clue in clue_data:
        total_loss += clue.weight * np.abs(clue.sims - clue.target_sim)

    clue_words = {word for word, _ in clues}
    order = np.argsort(total_loss)

    results: List[Tuple[str, float, float]] = []

    for idx in order:
        candidate = candidate_vocab[idx]
        if candidate in clue_words:
            continue

        # Report explainability metric: how far predicted ranks are from clue ranks.
        rank_errors = []
        for clue in clue_data:
            pred_rank = approx_rank_from_sorted(clue.sorted_sims, float(clue.sims[idx]))
            rank_errors.append(abs(pred_rank - clue.observed_rank))

        avg_rank_error = float(np.mean(rank_errors))
        score = 1.0 / (1.0 + float(total_loss[idx]))
        results.append((candidate, score, avg_rank_error))

        if len(results) >= topk:
            break

    return results


# -------------------- Output / CLI --------------------
def run_neighbor_query(word: str, topk: int, model) -> None:
    query = normalize_query(word)
    if not query:
        print("Please enter a non-empty word.")
        return

    try:
        results = top_k_similar(query, topk, model)
    except ValueError as exc:
        print(f"\nQuery failed: {exc}")
        return

    print(f"\nTop {topk} words similar to '{query}':")
    if not results:
        print("No similar words found.")
        return

    for rank, (candidate, similarity) in enumerate(results, start=1):
        print(f"{rank:2d}. {candidate:18s} similarity={similarity:.4f}")


def run_clue_solver(
    clues: Sequence[Tuple[str, int]],
    model,
    model_name: str,
    candidate_size: int,
    topk: int,
) -> None:
    candidate_vocab, candidate_vectors = load_or_create_candidate_matrix(
        model=model,
        model_name=model_name,
        candidate_size=candidate_size,
    )

    results = solve_from_clues(
        clues=clues,
        model=model,
        candidate_vocab=candidate_vocab,
        candidate_vectors=candidate_vectors,
        topk=topk,
    )

    print("\nSemantle clue mode")
    print(f"- model: {model_name}")
    print(f"- clues: {', '.join(f'{w}:{r}' for w, r in clues)}")
    print(f"- candidate pool: {len(candidate_vocab)} words")

    if not results:
        print("No candidate suggestions found.")
        return

    print(f"\nTop {topk} candidate secret words:")
    for rank, (word, score, avg_err) in enumerate(results, start=1):
        print(f"{rank:2d}. {word:18s} score={score:.4f}  avg_rank_error={avg_err:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantle helper with a single word2vec-based approach."
    )
    parser.add_argument("--word", type=str, help="Query word for neighbor suggestions.")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="How many results to return.")
    parser.add_argument(
        "--clue",
        action="append",
        default=[],
        help="Semantle clue in WORD:RANK format. Repeat for multiple clues.",
    )
    parser.add_argument(
        "--gensim-model",
        type=str,
        default=DEFAULT_GENSIM_MODEL,
        help=(
            "gensim model name. Default uses word2vec Google News vectors. "
            "For a smaller model use glove-wiki-gigaword-300."
        ),
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=DEFAULT_CANDIDATE_SIZE,
        help="Candidate pool size used by clue mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.candidate_size < 5_000:
        raise ValueError("--candidate-size should be at least 5000")

    if args.word and args.clue:
        raise ValueError("Use either --word or --clue mode, not both in the same run")

    model = load_gensim_model(args.gensim_model)

    if args.clue:
        clues = parse_clues(args.clue)
        run_clue_solver(
            clues=clues,
            model=model,
            model_name=args.gensim_model,
            candidate_size=args.candidate_size,
            topk=args.topk,
        )
        return

    if args.word:
        run_neighbor_query(args.word, args.topk, model)
        return

    print("Interactive mode (neighbor suggestions). Type a word. Type 'quit' to exit.")
    while True:
        query = input("> ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        run_neighbor_query(query, args.topk, model)


if __name__ == "__main__":
    main()
