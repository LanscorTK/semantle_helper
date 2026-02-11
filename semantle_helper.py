#!/usr/bin/env python3
"""Semantle helper using a single word2vec-based approach.

Features:
- Neighbor mode: suggest words semantically close to a query.
- Clue mode: combine Semantle clues (WORD:RANK) to suggest likely targets.
- Play mode: record guesses while playing and suggest next guesses.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
DEFAULT_PLAY_SUGGESTIONS = 5
STATUS_LABELS = {"cold", "tepid", "far"}
SEMANTLE_WEAKEST_RANK = 1
SEMANTLE_HOTTEST_RANK = 999


# -------------------- Data structures --------------------
@dataclass
class ClueData:
    word: str
    observed_rank_external: int
    observed_rank_internal: int
    sims: np.ndarray
    sorted_sims: np.ndarray
    target_sim: float
    weight: float


@dataclass
class ScoreAnchors:
    top2: float
    top10: float
    top1000: float


@dataclass
class GuessRecord:
    score: float
    provided_rank_external: int
    score_rank_external: int
    status: str | None = None


def display_rank_fields(record: GuessRecord) -> Tuple[str, str]:
    """Return display values for RANK/SCORE_RANK.

    Status-based entries represent guesses outside visible top-1000 ranking,
    so rank columns are shown as empty.
    """
    if record.status is not None:
        return "", ""
    return str(record.provided_rank_external), str(record.score_rank_external)


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


def semantle_rank_to_internal(external_rank: int, max_internal_rank: int) -> int:
    """Map Semantle proximity rank to internal candidate-rank space.

    Semantle convention used here:
    - rank 999: hottest non-answer word
    - rank 1: weakest word still inside Semantle's top-1000 list

    Internal convention:
    - rank 1: hottest candidate
    """
    if max_internal_rank <= 1:
        return 1

    clamped = int(np.clip(external_rank, SEMANTLE_WEAKEST_RANK, SEMANTLE_HOTTEST_RANK))
    ratio = (clamped - SEMANTLE_WEAKEST_RANK) / (SEMANTLE_HOTTEST_RANK - SEMANTLE_WEAKEST_RANK)

    # Log-scale projection: Semantle top-1000 rank range maps onto the full candidate pool.
    log_internal = np.log(float(max_internal_rank)) * (1.0 - ratio)
    internal = int(np.rint(np.exp(log_internal)))
    return int(np.clip(internal, 1, max_internal_rank))


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
    """Build/load candidate vectors used in clue and play mode.

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
    """Parse repeated WORD:RANK clues from CLI.

    Rank is Semantle proximity rank in [1, 999], where higher means closer.
    """
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

        if not (SEMANTLE_WEAKEST_RANK <= rank <= SEMANTLE_HOTTEST_RANK):
            raise ValueError(
                f"Rank must be between {SEMANTLE_WEAKEST_RANK} and {SEMANTLE_HOTTEST_RANK} in clue '{raw}'"
            )

        # Duplicate clue words add no information; ignore them.
        if word in seen:
            continue

        seen.add(word)
        clues.append((word, rank))

    if not clues:
        raise ValueError("No valid clues provided")

    return clues


# -------------------- Play mode score mapping --------------------
def validate_score_anchors(anchors: ScoreAnchors) -> None:
    """Require strict score ordering: rank999 > rank990 > rank1."""
    if not (anchors.top2 > anchors.top10 > anchors.top1000):
        raise ValueError(
            "Score anchors must satisfy: rank999_score > rank990_score > rank1_score"
        )


def prompt_float(prompt_text: str) -> float:
    while True:
        raw = input(prompt_text).strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def get_score_anchors_from_args_or_prompt(args: argparse.Namespace) -> ScoreAnchors:
    provided = [args.top2_score, args.top10_score, args.top1000_score]

    if any(value is not None for value in provided):
        if not all(value is not None for value in provided):
            raise ValueError(
                "If using score flags, provide all: --top999-score, --top990-score, --top1-score"
            )
        anchors = ScoreAnchors(
            top2=float(args.top2_score),
            top10=float(args.top10_score),
            top1000=float(args.top1000_score),
        )
        validate_score_anchors(anchors)
        return anchors

    print("Enter today's Semantle reference similarity scores:")
    anchors = ScoreAnchors(
        top2=prompt_float("- nearest non-answer word score (rank 999): "),
        top10=prompt_float("- 10th nearest word score (rank 990): "),
        top1000=prompt_float("- weakest ranked word score (rank 1): "),
    )
    validate_score_anchors(anchors)
    return anchors


def estimate_rank_from_score(score: float, anchors: ScoreAnchors, max_external_rank: int) -> int:
    """Map similarity score to Semantle proximity rank.

    External ranking convention used here:
    - rank 999: hottest non-answer word
    - rank 990: 10th-hottest non-answer word
    - rank 1: weakest word still inside Semantle's top-1000 list
    """
    log1 = float(np.log(1.0))
    log10 = float(np.log(10.0))
    log999 = float(np.log(999.0))

    s999 = anchors.top2
    s990 = anchors.top10
    s1 = anchors.top1000

    max_rank = min(max_external_rank, SEMANTLE_HOTTEST_RANK)

    # score==100 (exact answer) is handled by caller before this function.
    if score >= s999:
        return max_rank

    if score >= s990:
        t = (score - s999) / (s990 - s999)
        log_pos = log1 + t * (log10 - log1)
    elif score >= s1:
        t = (score - s990) / (s1 - s990)
        log_pos = log10 + t * (log999 - log10)
    else:
        slope = (log999 - log10) / (s1 - s990)
        log_pos = log999 + (score - s1) * slope

    position = int(np.rint(np.exp(log_pos)))
    rank = (SEMANTLE_HOTTEST_RANK + 1) - position
    return int(np.clip(rank, SEMANTLE_WEAKEST_RANK, max_rank))


def parse_rank_or_status(token: str) -> int | str:
    """Parse third field as numeric rank or status label."""
    try:
        return int(token)
    except ValueError:
        status = normalize_query(token).strip("()")
        if status not in STATUS_LABELS:
            raise ValueError("Ranking must be an integer or one of: cold, tepid, far")
        return status


def infer_rank_from_status(
    status: str,
    score: float,
    score_rank_external: int,
    max_external_rank: int,
) -> int:
    """Convert status (cold/tepid/far) into a coarse external rank constraint.

    Higher rank means semantically hotter.
    """
    status = normalize_query(status)

    if status == "tepid":
        # Tepid is usually closer than cold/far but still coarse.
        base_rank = max(140, min(score_rank_external, 420))
        if score <= 20:
            print("Note: status 'tepid' is usually above ~20 score, but accepted.")
    elif status == "cold":
        # Cold is weaker than tepid.
        base_rank = max(40, min(score_rank_external, 220))
        if score < 0:
            print("Note: status 'cold' is usually non-negative, but accepted.")
    else:  # far
        # Far is very weak signal and typically negative.
        base_rank = max(1, min(score_rank_external, 80))
        if score >= 0:
            print("Note: status 'far' is usually negative, but accepted.")

    max_rank = min(max_external_rank, SEMANTLE_HOTTEST_RANK)
    return int(np.clip(base_rank, SEMANTLE_WEAKEST_RANK, max_rank))


def infer_status_from_score(score: float) -> str:
    """Infer Semantle status label when rank is not shown."""
    if score < 0:
        return "far"
    if score > 20:
        return "tepid"
    return "cold"


# -------------------- Clue mode scoring math --------------------
def approx_rank_from_sorted(sorted_sims: np.ndarray, sim: float) -> int:
    """Approximate internal rank from similarity against sorted candidate similarities.

    Internal rank convention:
    - rank 1: hottest candidate (corresponds to external rank 999)
    """
    ge_count = int(len(sorted_sims) - np.searchsorted(sorted_sims, sim, side="left"))
    return max(1, ge_count - 1)


def build_clue_data(
    clues: Sequence[Tuple[str, int] | Tuple[str, int, float]],
    model,
    candidate_vectors: np.ndarray,
) -> List[ClueData]:
    """Convert external clue ranks into internal similarity bands for scoring.

    Optional clue weight multiplier can be provided as a 3rd tuple element.
    """
    clue_data: List[ClueData] = []
    n = len(candidate_vectors)

    for clue in clues:
        if len(clue) == 2:
            clue_word, observed_rank_external = clue
            clue_strength = 1.0
        else:
            clue_word, observed_rank_external, clue_strength = clue

        token = resolve_gensim_token(clue_word, model)
        if token is None:
            raise ValueError(f"Clue word '{clue_word}' is not in model vocabulary")

        clue_vec = unit_vector(model, token)
        sims = candidate_vectors @ clue_vec

        observed_rank_internal = semantle_rank_to_internal(observed_rank_external, n)

        # Convert observed internal rank to the similarity value at that rank.
        desc_index = min(max(observed_rank_internal, 1), n - 1)
        kth = n - 1 - desc_index
        target_sim = float(np.partition(sims.copy(), kth)[kth])

        # Hotter clues (smaller internal ranks) get larger influence in total loss.
        weight = (1.0 / np.log2(observed_rank_internal + 2.0)) * float(clue_strength)

        clue_data.append(
            ClueData(
                word=clue_word,
                observed_rank_external=observed_rank_external,
                observed_rank_internal=observed_rank_internal,
                sims=sims,
                sorted_sims=np.sort(sims),
                target_sim=target_sim,
                weight=weight,
            )
        )

    return clue_data


def solve_from_clues(
    clues: Sequence[Tuple[str, int] | Tuple[str, int, float]],
    model,
    candidate_vocab: Sequence[str],
    candidate_vectors: np.ndarray,
    topk: int,
) -> List[Tuple[str, float, float]]:
    """Suggest likely secret words from Semantle clues.

    Input ranks are Semantle external ranks. Returns:
    (candidate_word, score, avg_rank_error)
    """
    clue_data = build_clue_data(clues, model, candidate_vectors)

    total_loss = np.zeros(len(candidate_vocab), dtype=np.float32)

    # Candidate score = weighted distance from each clue's implied target similarity.
    for clue in clue_data:
        total_loss += clue.weight * np.abs(clue.sims - clue.target_sim)

    clue_words = {clue[0] for clue in clues}
    order = np.argsort(total_loss)

    results: List[Tuple[str, float, float]] = []

    for idx in order:
        candidate = candidate_vocab[idx]
        if candidate in clue_words:
            continue

        # Report explainability metric in internal-rank space.
        rank_errors = []
        for clue in clue_data:
            pred_rank_internal = approx_rank_from_sorted(clue.sorted_sims, float(clue.sims[idx]))
            rank_errors.append(abs(pred_rank_internal - clue.observed_rank_internal))

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
    print(f"- clues (external ranks): {', '.join(f'{w}:{r}' for w, r in clues)}")
    print(f"- candidate pool: {len(candidate_vocab)} words")

    if not results:
        print("No candidate suggestions found.")
        return

    print(f"\nTop {topk} candidate secret words:")
    for rank, (word, score, avg_err) in enumerate(results, start=1):
        print(f"{rank:2d}. {word:18s} score={score:.4f}  avg_rank_error={avg_err:.1f}")


def print_play_suggestions(
    records: Dict[str, GuessRecord],
    model,
    candidate_vocab: Sequence[str],
    candidate_vectors: np.ndarray,
    suggestion_count: int,
) -> None:
    if not records:
        print("No guesses recorded yet.")
        return

    clues: List[Tuple[str, int, float]] = []
    for word, rec in records.items():
        if rec.status is None:
            # Numeric rank is a stronger clue.
            clues.append((word, rec.provided_rank_external, 1.0))
            if rec.score_rank_external != rec.provided_rank_external:
                clues.append((word, rec.score_rank_external, 0.6))
        else:
            # Status-based rank is coarse, so keep weaker weights.
            clues.append((word, rec.provided_rank_external, 0.45))
            if rec.score_rank_external != rec.provided_rank_external:
                clues.append((word, rec.score_rank_external, 0.35))

    raw_results = solve_from_clues(
        clues=clues,
        model=model,
        candidate_vocab=candidate_vocab,
        candidate_vectors=candidate_vectors,
        topk=max(20, suggestion_count + 10),
    )

    print(f"\nNext {suggestion_count} suggested guesses:")
    shown = 0
    for word, score, avg_err in raw_results:
        if word in records:
            continue
        shown += 1
        print(f"{shown:2d}. {word:18s} score={score:.4f}  avg_rank_error={avg_err:.1f}")
        if shown >= suggestion_count:
            break

    if shown == 0:
        print("No suggestions available with current clues.")


def print_random_words(
    candidate_vocab: Sequence[str],
    records: Dict[str, GuessRecord],
    suggestion_count: int,
) -> None:
    """Print random candidate words for exploration during play mode."""
    if suggestion_count <= 0:
        print("Random count must be > 0.")
        return

    available = [word for word in candidate_vocab if word not in records]
    if not available:
        print("No random words available (all candidates already recorded).")
        return

    pick_count = min(suggestion_count, len(available))
    picks = random.sample(available, k=pick_count)

    print(f"\nRandom {pick_count} words:")
    for idx, word in enumerate(picks, start=1):
        print(f"{idx:2d}. {word}")


def print_play_records(records: Dict[str, GuessRecord]) -> None:
    if not records:
        print("No guesses recorded yet.")
        return

    rows = []
    for idx, (word, rec) in enumerate(
        sorted(records.items(), key=lambda item: item[1].provided_rank_external, reverse=True), start=1
    ):
        rank_display, score_rank_display = display_rank_fields(rec)
        rows.append(
            (
                str(idx),
                word,
                f"{rec.score:.4f}",
                rank_display,
                score_rank_display,
                rec.status if rec.status else "-",
            )
        )

    headers = ("#", "WORD", "SCORE", "RANK", "SCORE_RANK", "STATUS")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt(row: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

    print("\nRecorded guesses:")
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def parse_play_record(raw: str) -> Tuple[str, float, int | str] | None:
    """Parse '<word> <score> <rank_or_status>'; return None if format does not match."""
    parts = raw.split()
    if len(parts) != 3:
        return None

    word = normalize_query(parts[0])
    if not is_clean_word(word):
        raise ValueError("Invalid word format. Use alphabetic words only.")

    try:
        score = float(parts[1])
    except ValueError as exc:
        raise ValueError("Invalid score. Use a numeric similarity score.") from exc

    rank_or_status = parse_rank_or_status(parts[2])
    return word, score, rank_or_status


def build_record_from_input(
    word: str,
    score: float,
    rank_or_status: int | str,
    anchors: ScoreAnchors,
    max_external_rank: int,
) -> GuessRecord:
    """Build GuessRecord from numeric rank or status label."""
    score_rank_external = estimate_rank_from_score(
        score=score,
        anchors=anchors,
        max_external_rank=max_external_rank,
    )

    if isinstance(rank_or_status, int):
        if not (SEMANTLE_WEAKEST_RANK <= rank_or_status <= SEMANTLE_HOTTEST_RANK):
            raise ValueError("Invalid ranking. Use an integer rank between 1 and 999.")

        provided_rank_external = min(rank_or_status, min(max_external_rank, SEMANTLE_HOTTEST_RANK))
        return GuessRecord(
            score=score,
            provided_rank_external=provided_rank_external,
            score_rank_external=score_rank_external,
            status=None,
        )

    status = normalize_query(rank_or_status)
    provided_rank_external = infer_rank_from_status(
        status=status,
        score=score,
        score_rank_external=score_rank_external,
        max_external_rank=max_external_rank,
    )

    return GuessRecord(
        score=score,
        provided_rank_external=provided_rank_external,
        score_rank_external=score_rank_external,
        status=status,
    )


def run_play_mode(
    model,
    model_name: str,
    candidate_size: int,
    anchors: ScoreAnchors,
    suggestion_count: int,
) -> None:
    candidate_vocab, candidate_vectors = load_or_create_candidate_matrix(
        model=model,
        model_name=model_name,
        candidate_size=candidate_size,
    )

    max_external_rank = SEMANTLE_HOTTEST_RANK
    records: Dict[str, GuessRecord] = {}

    print("\nPlay mode started.")
    print(
        "Enter guesses as: <word> <score> (auto status) or <word> <score> <rank>. "
        "Commands: suggest, random, list, amend <old> <new> <score> [rank], remove <word>, help, quit"
    )
    print("Auto status rules when no rank is shown: far(score<0), cold(0..20), tepid(>20)")
    print("Type a single word to see its top 10 similar words.")
    print("If a guess has score 100, type '<word> 100' or '<word> 100 <rank>' to end.")
    print(
        "Ranking note: higher rank means closer; rank 999 is hottest non-answer, rank 1 is the edge of top-1000."
    )
    print(
        f"Using anchors: rank999={anchors.top2}, rank990={anchors.top10}, rank1={anchors.top1000}"
    )

    while True:
        raw = input("play> ").strip()
        if not raw:
            continue

        lower = raw.lower()

        if lower in {"quit", "exit", "q"}:
            break

        if lower in {"help", "?"}:
            print("Commands:")
            print("- <word> <score>                   add/update guess (auto status) and get suggestions")
            print("- <word> <score> <rank>            add/update guess with explicit numeric rank")
            print("- <word>                           show top 10 similar words")
            print("- <word> 100                       mark solved and end play mode")
            print("- amend <old> <new> <score> [rank] fix typos/mistakes")
            print("  auto status rules: far(score<0), cold(0..20), tepid(>20)")
            print("  rank note: higher rank is hotter; valid explicit rank is 1..999")
            print("- suggest [n]                      show clue-based suggestions (default 5)")
            print("- random [n]                       show random words (default 5)")
            print("- list                             show recorded guesses")
            print("- remove <word>                    remove a recorded guess")
            print("- quit                             exit play mode")
            continue

        if lower == "suggest" or lower.startswith("suggest "):
            suggest_parts = raw.split()
            if len(suggest_parts) == 1:
                suggest_n = suggestion_count
            elif len(suggest_parts) == 2:
                try:
                    suggest_n = int(suggest_parts[1])
                except ValueError:
                    print("Usage: suggest [n]")
                    continue
                if suggest_n <= 0:
                    print("Usage: suggest [n] with n > 0")
                    continue
            else:
                print("Usage: suggest [n]")
                continue

            print_play_suggestions(
                records=records,
                model=model,
                candidate_vocab=candidate_vocab,
                candidate_vectors=candidate_vectors,
                suggestion_count=suggest_n,
            )
            continue

        if lower == "random" or lower.startswith("random "):
            random_parts = raw.split()
            if len(random_parts) == 1:
                random_n = suggestion_count
            elif len(random_parts) == 2:
                try:
                    random_n = int(random_parts[1])
                except ValueError:
                    print("Usage: random [n]")
                    continue
                if random_n <= 0:
                    print("Usage: random [n] with n > 0")
                    continue
            else:
                print("Usage: random [n]")
                continue

            print_random_words(
                candidate_vocab=candidate_vocab,
                records=records,
                suggestion_count=random_n,
            )
            continue

        if lower == "list":
            print_play_records(records)
            continue

        if lower.startswith("remove "):
            to_remove = normalize_query(raw.split(maxsplit=1)[1])
            if to_remove in records:
                del records[to_remove]
                print(f"Removed '{to_remove}'.")
            else:
                print(f"'{to_remove}' not found in records.")
            continue

        if lower.startswith("amend "):
            parts = raw.split()
            if len(parts) not in {4, 5}:
                print("Usage: amend <old_word> <new_word> <score> [rank]")
                continue

            old_word = normalize_query(parts[1])
            new_word = normalize_query(parts[2])

            if old_word not in records:
                print(f"'{old_word}' is not in recorded guesses.")
                continue
            if not is_clean_word(new_word):
                print("Invalid new word format. Use alphabetic words only.")
                continue
            if resolve_gensim_token(new_word, model) is None:
                print(f"'{new_word}' is not in the selected model vocabulary.")
                continue

            try:
                score = float(parts[3])
            except ValueError:
                print("Invalid score. Use a numeric similarity score.")
                continue

            if np.isclose(score, 100.0):
                print(f"Solved! '{new_word}' has similarity 100. Ending play mode.")
                break

            try:
                if len(parts) == 5:
                    rank_or_status = parse_rank_or_status(parts[4])
                else:
                    rank_or_status = infer_status_from_score(score)

                new_record = build_record_from_input(
                    word=new_word,
                    score=score,
                    rank_or_status=rank_or_status,
                    anchors=anchors,
                    max_external_rank=max_external_rank,
                )
            except ValueError as exc:
                print(str(exc))
                continue

            if old_word != new_word and new_word in records:
                print(f"Note: '{new_word}' already exists and will be replaced.")
                del records[new_word]

            del records[old_word]
            records[new_word] = new_record

            rank_display, score_rank_display = display_rank_fields(new_record)
            status_suffix = f" status={new_record.status}" if new_record.status else ""
            print(
                f"Amended: {old_word} -> {new_word} score={new_record.score:.4f} "
                f"rank={rank_display} score_rank={score_rank_display}{status_suffix}"
            )

            print_play_suggestions(
                records=records,
                model=model,
                candidate_vocab=candidate_vocab,
                candidate_vectors=candidate_vectors,
                suggestion_count=suggestion_count,
            )
            continue

        parts = raw.split()

        if len(parts) == 1:
            probe = normalize_query(parts[0])
            if not is_clean_word(probe):
                print("Invalid input. Use: <word> <score> or <word> <score> <rank> (or type 'help').")
                continue
            run_neighbor_query(probe, 10, model)
            continue

        if len(parts) == 2:
            word = normalize_query(parts[0])
            if not is_clean_word(word):
                print("Invalid word format. Use alphabetic words only.")
                continue
            if resolve_gensim_token(word, model) is None:
                print(f"'{word}' is not in the selected model vocabulary.")
                continue

            try:
                score = float(parts[1])
            except ValueError:
                print("Invalid score. Use a numeric similarity score.")
                continue

            if np.isclose(score, 100.0):
                print(f"Solved! '{word}' has similarity 100. Ending play mode.")
                break

            auto_status = infer_status_from_score(score)
            record = build_record_from_input(
                word=word,
                score=score,
                rank_or_status=auto_status,
                anchors=anchors,
                max_external_rank=max_external_rank,
            )

            records[word] = record
            rank_display, score_rank_display = display_rank_fields(record)
            print(
                f"Recorded: {word} score={record.score:.4f} rank={rank_display} "
                f"score_rank={score_rank_display} status={record.status}"
            )

            print_play_suggestions(
                records=records,
                model=model,
                candidate_vocab=candidate_vocab,
                candidate_vectors=candidate_vectors,
                suggestion_count=suggestion_count,
            )
            continue

        try:
            parsed = parse_play_record(raw)
        except ValueError as exc:
            print(str(exc))
            continue

        if parsed is None:
            print("Invalid input. Use: <word> <score> or <word> <score> <rank> (or type 'help').")
            continue

        word, score, rank_or_status = parsed

        if resolve_gensim_token(word, model) is None:
            print(f"'{word}' is not in the selected model vocabulary.")
            continue

        if np.isclose(score, 100.0):
            print(f"Solved! '{word}' has similarity 100. Ending play mode.")
            break

        try:
            record = build_record_from_input(
                word=word,
                score=score,
                rank_or_status=rank_or_status,
                anchors=anchors,
                max_external_rank=max_external_rank,
            )
        except ValueError as exc:
            print(str(exc))
            continue

        records[word] = record

        rank_display, score_rank_display = display_rank_fields(record)
        status_suffix = f" status={record.status}" if record.status else ""
        print(
            f"Recorded: {word} score={record.score:.4f} rank={rank_display} "
            f"score_rank={score_rank_display}{status_suffix}"
        )

        print_play_suggestions(
            records=records,
            model=model,
            candidate_vocab=candidate_vocab,
            candidate_vectors=candidate_vectors,
            suggestion_count=suggestion_count,
        )


# -------------------- Argument parsing --------------------
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
        help="Semantle clue in WORD:RANK format (rank 1..999, higher is closer). Repeat for multiple clues.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Interactive play mode with score anchors and guess logging.",
    )
    parser.add_argument(
        "--suggestions",
        type=int,
        default=DEFAULT_PLAY_SUGGESTIONS,
        help="Number of next-guess suggestions in play mode.",
    )
    parser.add_argument(
        "--top999-score",
        "--top2-score",
        dest="top2_score",
        type=float,
        help="Today's nearest non-answer score (Semantle rank 999).",
    )
    parser.add_argument(
        "--top990-score",
        "--top10-score",
        dest="top10_score",
        type=float,
        help="Today's 10th-nearest score (Semantle rank 990).",
    )
    parser.add_argument(
        "--top1-score",
        "--top1000-score",
        dest="top1000_score",
        type=float,
        help="Today's weakest ranked score still in top-1000 (Semantle rank 1).",
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
        help="Candidate pool size used by clue/play mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.suggestions <= 0:
        raise ValueError("--suggestions must be > 0")
    if args.candidate_size < 5_000:
        raise ValueError("--candidate-size should be at least 5000")

    if args.play and (args.word or args.clue):
        raise ValueError("Use --play by itself (do not combine with --word or --clue)")
    if args.word and args.clue:
        raise ValueError("Use either --word or --clue mode, not both in the same run")

    model = load_gensim_model(args.gensim_model)

    if args.play:
        anchors = get_score_anchors_from_args_or_prompt(args)
        run_play_mode(
            model=model,
            model_name=args.gensim_model,
            candidate_size=args.candidate_size,
            anchors=anchors,
            suggestion_count=args.suggestions,
        )
        return

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
