"""
Data loading and sampling utilities for activation plateau analysis.

Provides FineWeb loading and covariance-matched random activation sampling.
"""

import time

import torch
import numpy as np
from datasets import load_dataset


def generate_random_token_sequences(n, vocab_size, seq_len, seed=42):
    """
    Generate random token sequences from a uniform distribution over the vocabulary.

    Useful for models trained on synthetic data (e.g., CSP/circuitgpt) where
    natural text is not appropriate.

    Args:
        n: Number of sequences to generate
        vocab_size: Size of token vocabulary
        seq_len: Length of each sequence
        seed: Random seed

    Returns:
        token_ids_list: list of n tensors, each [seq_len]
    """
    torch.manual_seed(seed)
    token_ids_list = [torch.randint(0, vocab_size, (seq_len,)) for _ in range(n)]
    print(f"Generated {n} random token sequences (vocab_size={vocab_size}, seq_len={seq_len})")
    return token_ids_list


def generate_chess_games(n, seq_len, seed=42):
    """
    Generate random legal chess games as character-level token sequences.

    Uses the nanoGPT chess model's character-level vocabulary (32 tokens).
    Games are in PGN-like format starting with ';' delimiter.

    Args:
        n: Number of sequences to generate
        seq_len: Length of each sequence in characters/tokens
        seed: Random seed

    Returns:
        token_ids_list: list of n tensors, each [seq_len]
    """
    import chess
    import random
    random.seed(seed)

    # Character-level vocab from meta.pkl
    stoi = {' ': 0, '#': 1, '+': 2, '-': 3, '.': 4, '0': 5, '1': 6, '2': 7,
            '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14,
            ';': 15, '=': 16, 'B': 17, 'K': 18, 'N': 19, 'O': 20, 'Q': 21,
            'R': 22, 'a': 23, 'b': 24, 'c': 25, 'd': 26, 'e': 27, 'f': 28,
            'g': 29, 'h': 30, 'x': 31}

    def game_to_pgn(board):
        """Convert a board's move stack to PGN string."""
        pgn = ";"
        temp = chess.Board()
        for i, move in enumerate(board.move_stack):
            if i % 2 == 0:
                pgn += f"{i//2 + 1}."
            pgn += temp.san(move) + " "
            temp.push(move)
        return pgn

    token_ids_list = []
    attempts = 0
    while len(token_ids_list) < n and attempts < n * 20:
        attempts += 1
        board = chess.Board()
        # Play random moves until game ends or we have enough characters
        max_moves = seq_len  # generous upper bound
        for _ in range(max_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(random.choice(legal))

        pgn = game_to_pgn(board)
        # Check all characters are in vocab
        if not all(c in stoi for c in pgn):
            continue
        if len(pgn) < seq_len:
            continue

        tokens = torch.tensor([stoi[c] for c in pgn[:seq_len]], dtype=torch.long)
        token_ids_list.append(tokens)

    print(f"Generated {len(token_ids_list)} chess game sequences (each {seq_len} chars)")
    return token_ids_list


def load_python_code_fixed_length(n, tokenizer, seq_len=5, seed=42):
    """
    Load Python code from CodeParrot and tokenize to fixed-length sequences.

    Uses the CSP model's custom tokenizer. Each sample is a seq_len-token
    subsequence drawn from real Python files.

    Args:
        n: Number of sequences to return
        tokenizer: CSP tokenizer (from AutoTokenizer with trust_remote_code)
        seq_len: Exact token length for each sequence
        seed: Random seed for dataset shuffling

    Returns:
        token_ids_list: list of n tensors, each [seq_len]
    """
    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    token_ids_list = []

    for sample in dataset:
        code = sample['content']
        ids = tokenizer.encode(code)
        # Extract as many non-overlapping seq_len chunks as possible
        for start in range(0, len(ids) - seq_len + 1, seq_len):
            token_ids_list.append(torch.tensor(ids[start:start + seq_len], dtype=torch.long))
            if len(token_ids_list) >= n:
                break
        if len(token_ids_list) >= n:
            break

    print(f"Loaded {len(token_ids_list)} Python code sequences (each {seq_len} tokens)")
    return token_ids_list


def load_fineweb_samples(n, tokenizer, min_tokens=10, max_tokens=512, seed=42):
    """
    Load and tokenize samples from FineWeb.

    Args:
        n: Number of samples to return
        tokenizer: GPT-2 tokenizer
        min_tokens: Minimum token length (filter shorter texts)
        max_tokens: Maximum token length (truncate longer texts)
        seed: Random seed for shuffling

    Returns:
        texts: list of n strings
        token_counts: list of token counts per text
    """
    # Load FineWeb-Edu sample (smaller, cleaner subset)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    # Shuffle with seed
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    texts = []
    token_counts = []

    for sample in dataset:
        text = sample['text']

        # Tokenize to check length
        tokens = tokenizer(text, truncation=True, max_length=max_tokens)
        n_tokens = len(tokens['input_ids'])

        if n_tokens >= min_tokens:
            texts.append(text)
            token_counts.append(n_tokens)

        if len(texts) >= n:
            break

    print(f"Loaded {len(texts)} FineWeb samples")
    print(f"Token counts: min={min(token_counts)}, max={max(token_counts)}, "
          f"mean={np.mean(token_counts):.1f}")

    return texts, token_counts


def load_fineweb_fixed_length(n, tokenizer, seq_len=5, seed=42, cache_dir="/workspace/fineweb_cache",
                              add_special_tokens=True):
    """
    Load FineWeb samples tokenized to exactly seq_len tokens.

    Checks for a cached file first (on network volume) to avoid re-streaming.
    Truncates longer texts to seq_len. Skips texts shorter than seq_len.

    Args:
        n: Number of samples to return
        tokenizer: GPT-2 tokenizer
        seq_len: Exact token length for all sequences
        seed: Random seed for shuffling
        cache_dir: Directory to check for cached token files

    Returns:
        token_ids_list: list of n tensors, each [seq_len]
    """
    import os
    cache_file = os.path.join(cache_dir, f"fineweb_n{n}_seq{seq_len}_seed{seed}.pt")
    if os.path.exists(cache_file):
        token_ids_list = torch.load(cache_file, weights_only=True)
        print(f"Loaded {len(token_ids_list)} cached FineWeb samples from {cache_file}")
        return token_ids_list

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    token_ids_list = []
    batch_size = 1000  # tokenize in batches for speed
    text_batch = []
    t0 = time.time()

    for sample in dataset:
        text_batch.append(sample['text'])

        if len(text_batch) >= batch_size:
            # Batch tokenize
            batch_tokens = tokenizer(text_batch, truncation=True, max_length=seq_len,
                                     add_special_tokens=add_special_tokens,
                                     padding=False)
            for ids in batch_tokens['input_ids']:
                if len(ids) >= seq_len:
                    token_ids_list.append(torch.tensor(ids[:seq_len]))
                    if len(token_ids_list) >= n:
                        break
            text_batch = []

            if len(token_ids_list) >= n:
                break

            if len(token_ids_list) % 100000 == 0 and len(token_ids_list) > 0:
                rate = len(token_ids_list) / (time.time() - t0)
                remaining = (n - len(token_ids_list)) / rate
                print(f"  Collected {len(token_ids_list)}/{n} sequences "
                      f"({rate:.0f}/s, ~{remaining:.0f}s remaining)...")

    # Process any remaining texts
    if text_batch and len(token_ids_list) < n:
        batch_tokens = tokenizer(text_batch, truncation=True, max_length=seq_len,
                                 add_special_tokens=add_special_tokens,
                                 padding=False)
        for ids in batch_tokens['input_ids']:
            if len(ids) >= seq_len:
                token_ids_list.append(torch.tensor(ids[:seq_len]))
                if len(token_ids_list) >= n:
                    break

    elapsed = time.time() - t0
    print(f"Loaded {len(token_ids_list)} FineWeb samples (each {seq_len} tokens) in {elapsed:.0f}s")

    # Save to cache for future use
    if cache_dir and len(token_ids_list) == n:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(token_ids_list, cache_file)
        print(f"Cached to {cache_file}")

    return token_ids_list


def sample_random_activations_v2(real_activations, n, mode='full', seed=42, device='cpu'):
    """
    Sample random activations from a Gaussian fitted to real activations.

    Supports both diagonal and full covariance. Works with concatenated
    all-position activations (seq_len * hidden_dim dimensional).
    Uses torch for covariance computation (GPU-accelerated when device='cuda').

    Args:
        real_activations: [N, D] tensor (D = seq_len * hidden_dim)
        n: Number of random samples to generate
        mode: 'diagonal' or 'full'
        seed: Random seed
        device: torch device for computation ('cuda' to use GPU)

    Returns:
        random_activations: [n, D] tensor (CPU)
        stats: dict with fitting info
    """
    torch.manual_seed(seed)

    if not isinstance(real_activations, torch.Tensor):
        real_activations = torch.tensor(real_activations, dtype=torch.float32)

    data = real_activations.to(device)
    N, D = data.shape
    mean = data.mean(dim=0)

    if mode == 'diagonal':
        var = data.var(dim=0, unbiased=True)
        print(f"Diagonal covariance: D={D}, variance range=[{var.min().item():.4f}, {var.max().item():.4f}]")
        # Sample: each entry independently from N(mean_i, var_i)
        samples = torch.randn(n, D, device=device) * var.sqrt() + mean
        stats = {'mode': 'diagonal'}

    elif mode == 'full':
        if N <= D:
            print(f"Warning: N={N} <= D={D}, covariance will be rank-deficient")
        # Compute covariance on GPU: cov = (X - mean)^T (X - mean) / (N - 1)
        centered = data - mean.unsqueeze(0)
        print(f"Computing covariance matrix ({D}x{D}) from {N} samples...")
        cov = (centered.T @ centered) / (N - 1)
        print(f"Computing Cholesky decomposition...")
        # Cholesky for sampling: sample = mean + L @ z, where z ~ N(0, I)
        try:
            L = torch.linalg.cholesky(cov)
        except torch.linalg.LinAlgError:
            # Add regularization, increasing until Cholesky succeeds
            eye = torch.eye(D, device=device)
            for scale in [1e-6, 1e-5, 1e-4, 1e-3]:
                reg = scale * torch.trace(cov) / D
                print(f"Cholesky failed, trying regularization eps={reg.item():.2e}")
                try:
                    L = torch.linalg.cholesky(cov + reg * eye)
                    break
                except torch.linalg.LinAlgError:
                    continue
            else:
                raise RuntimeError("Cholesky failed even with large regularization")
        z = torch.randn(n, D, device=device)
        samples = mean.unsqueeze(0) + z @ L.T
        if samples.isnan().any():
            print("Cholesky sampling produced NaN, falling back to SVD...")
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = eigvals.clamp(min=0)
            samples = mean.unsqueeze(0) + z @ (eigvecs * eigvals.sqrt().unsqueeze(0)).T
        rank = torch.linalg.matrix_rank(cov).item()
        print(f"Full covariance: D={D}, rank={rank}, N={N}")
        stats = {'mode': 'full', 'rank': rank}

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'diagonal' or 'full'.")

    random_activations = samples.cpu()
    print(f"Sampled {n} random activations (mode={mode})")

    return random_activations, stats


def sample_random_activations(real_activations, n, seed=42):
    """
    Sample random activations from a Gaussian fitted to real activations.

    Fits a multivariate Gaussian with full covariance to the real activations,
    then samples n vectors from it. Requires len(real_activations) > hidden_dim
    for full-rank covariance.

    Args:
        real_activations: [N, hidden_dim] tensor of real activations (N > d)
        n: Number of random samples to generate
        seed: Random seed

    Returns:
        random_activations: [n, hidden_dim] tensor
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(real_activations, torch.Tensor):
        real_activations = real_activations.numpy()

    N, d = real_activations.shape

    if N <= d:
        raise ValueError(
            f"Need more samples than dimensions for full-rank covariance. "
            f"Got {N} samples, {d} dimensions."
        )

    # Fit Gaussian
    mean = np.mean(real_activations, axis=0)
    cov = np.cov(real_activations, rowvar=False)

    # Check rank
    rank = np.linalg.matrix_rank(cov)
    print(f"Covariance matrix: shape={cov.shape}, rank={rank}")

    if rank < d:
        print(f"Warning: covariance is rank-deficient ({rank} < {d})")

    # Sample from multivariate Gaussian
    random_np = np.random.multivariate_normal(mean, cov, size=n)
    random_activations = torch.tensor(random_np, dtype=torch.float32)

    print(f"Sampled {n} random activations from fitted Gaussian")

    return random_activations
