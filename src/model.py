"""
Model loading and layer-wise forward pass utilities.

Supports GPT-2 family and Llama-architecture models (Qwen3, etc.).
Uses HuggingFace transformers directly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# GPT2LMHeadModel + GPT2Tokenizer imported lazily inside the gpt2 branch below
# (importing them at top level pulls in transformers' vision deps via lazy loader,
# which fails when torch/torchvision versions don't match)


# Short names → HuggingFace model IDs
MODEL_ALIASES = {
    'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
    'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
    'qwen3-4b': 'Qwen/Qwen3-4B',
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'gemma-1-2b': 'google/gemma-2b',
    'gemma-2-2b': 'google/gemma-2-2b',
    'gemma-2-9b': 'google/gemma-2-9b',
    'gemma-3-270m': 'google/gemma-3-270m',
    'gemma-3-1b': 'google/gemma-3-1b-pt',
    'gemma-3-4b': 'google/gemma-3-4b-pt',
    'gemma-3-12b': 'google/gemma-3-12b-pt',
    'csp': 'csp_yolo2',
    'csp_yolo1': 'csp_yolo1',
    'csp_hf': 'openai/circuit-sparsity',
    'dense_csp_1x': 'dense1_1x',
    'dense_csp_2x': 'dense1_2x',
    'dense_csp_4x': 'dense1_4x',
    'deepseek-r1-distill-qwen-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B',
    'llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
    'phi-1.5': 'microsoft/phi-1_5',
    'pythia-1.4b': 'EleutherAI/pythia-1.4b',
    'olmo-1b': 'allenai/OLMo-1B-hf',
    'olmo-7b': 'allenai/OLMo-7B-0724-hf',
    'mistral-7b-v0.3': 'mistralai/Mistral-7B-v0.3',
    'aya-expanse-8b': 'CohereForAI/aya-expanse-8b',
    'yi-1.5-9b': '01-ai/Yi-1.5-9B',
    'tinystories-28m': 'roneneldan/TinyStories-28M',
    'chess_16L': 'lichess_16layers_ckpt_no_optimizer.pt',
    'chess_8L': 'lichess_8layers_ckpt_no_optimizer.pt',
}

# Multimodal Gemma 3 models (4B+) need Gemma3ForCausalLM to extract the text decoder
GEMMA3_MULTIMODAL = {'google/gemma-3-4b-pt', 'google/gemma-3-12b-pt', 'google/gemma-3-27b-pt'}


def _get_blocks(model):
    """Get the list of transformer blocks."""
    model_type = getattr(model, '_plateau_info', {}).get('model_type', '')
    if model_type == 'circuitgpt':
        return model.circuit_model.transformer.h
    if hasattr(model, 'transformer'):  # GPT-2
        return model.transformer.h
    elif hasattr(model, 'gpt_neox'):  # GPT-NeoX (Pythia)
        return model.gpt_neox.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):  # Llama/Qwen
        return model.model.layers
    raise ValueError(f"Unknown model architecture: {type(model)}")


def _get_final_norm(model):
    """Get the final layer norm."""
    model_type = getattr(model, '_plateau_info', {}).get('model_type', '')
    if model_type == 'circuitgpt':
        return model.circuit_model.transformer.ln_f
    if hasattr(model, 'transformer'):  # GPT-2
        return model.transformer.ln_f
    elif hasattr(model, 'gpt_neox'):  # GPT-NeoX (Pythia)
        return model.gpt_neox.final_layer_norm
    elif hasattr(model, 'model') and hasattr(model.model, 'norm'):  # Llama/Qwen
        return model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):  # Phi
        return model.model.final_layernorm
    raise ValueError(f"Unknown model architecture: {type(model)}")


def _get_lm_head(model):
    """Get the language model head (unembedding)."""
    model_type = getattr(model, '_plateau_info', {}).get('model_type', '')
    if model_type == 'circuitgpt':
        return model.circuit_model.lm_head
    return model.lm_head


def _is_rope_model(model):
    """True for models that need explicit position_embeddings (Llama/Qwen/Gemma)."""
    return hasattr(model, '_plateau_info') and model._plateau_info['model_type'] in ('llama', 'gemma3')


def _get_position_embeddings(model, hidden_states):
    """Compute position embeddings for RoPE models.

    Args:
        model: Llama/Qwen/Gemma model
        hidden_states: [batch, seq_len, d_model] - used to infer seq_len and device

    Returns:
        For Gemma 3: ((cos_global, sin_global), (cos_local, sin_local))
        For others: (cos, sin) tuple
    """
    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    if model._plateau_info['model_type'] == 'gemma3':
        global_emb = model.model.rotary_emb(hidden_states, position_ids)
        local_emb = model.model.rotary_emb_local(hidden_states, position_ids)
        return (global_emb, local_emb)
    if hasattr(model, 'gpt_neox'):  # GPT-NeoX (Pythia)
        return model.gpt_neox.rotary_emb(hidden_states, position_ids)
    if hasattr(model.model, 'rotary_emb'):
        return model.model.rotary_emb(hidden_states, position_ids)
    # Fallback: rotary_emb lives per-layer (e.g. Gemma2 in transformers ≤4.44)
    return model.model.layers[0].self_attn.rotary_emb(hidden_states, position_ids)


def _is_cat_pos_emb_model(model):
    """True for models that concatenate position embeddings per-block (circuitgpt with d_pos_emb)."""
    if not (hasattr(model, '_plateau_info') and model._plateau_info['model_type'] == 'circuitgpt'):
        return False
    # Only models with d_pos_emb set use BlockCatPosEmb; dense baselines use regular Block
    return getattr(model.circuit_model.config, 'd_pos_emb', None) is not None


def _get_cat_pos_emb(model, seq_len):
    """Get concatenated position embeddings for circuitgpt.

    Returns:
        pos_emb: [1, seq_len, 32] tensor
    """
    return model.circuit_model.transformer.wpe.weight[:seq_len].unsqueeze(0)


def _run_block(block, hidden, model, pos_emb=None):
    """Run a single transformer block, handling different architectures.

    GPT-2 blocks return a tuple (hidden_states, ...).
    Llama/Qwen/Gemma blocks return a plain tensor.
    """
    model_type = model._plateau_info['model_type']
    if model_type == 'gemma3':
        global_emb, local_emb = pos_emb
        result = block(hidden, position_embeddings_global=global_emb,
                       position_embeddings_local=local_emb)
    elif model_type == 'llama':
        # position_embeddings kwarg available in transformers ≥4.47;
        # older versions compute RoPE internally and need position_ids instead
        if not hasattr(model, '_uses_position_embeddings'):
            import inspect
            model._uses_position_embeddings = (
                'position_embeddings' in inspect.signature(block.forward).parameters)
        if model._uses_position_embeddings:
            result = block(hidden, position_embeddings=pos_emb)
        else:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            result = block(hidden, position_ids=position_ids)
    elif model_type == 'circuitgpt' and pos_emb is not None:
        # BlockCatPosEmb.forward(x, pos_emb_to_cat)
        result = block(hidden, pos_emb)
    else:
        result = block(hidden)
    # Unpack tuple if needed (GPT-2 style)
    if isinstance(result, tuple):
        return result[0]
    return result


def load_model(model_name='gpt2', device=None, dtype=None):
    """
    Load model and tokenizer.

    Args:
        model_name: Short name or HuggingFace model name.
            GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            Qwen: 'qwen3-0.6b'
        device: torch device (auto-detected if None)
        dtype: torch dtype for model weights (e.g. torch.bfloat16).
            None = float32 for most models, bfloat16 for Gemma3 multimodal.

    Returns:
        model, tokenizer, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_name = MODEL_ALIASES.get(model_name, model_name)

    # Detect architecture type
    is_gpt2 = model_name.startswith('gpt2')
    is_chess = model_name.startswith('chess')

    is_csp = model_name == 'csp_hf'  # HF loading path (trust_remote_code)
    is_blob_csp = model_name.startswith('dense_csp') or model_name in ('csp_yolo1', 'csp')

    if is_chess:
        # nanoGPT chess model (adamkarvonen/chess_llms) loaded into HF GPT-2
        from huggingface_hub import hf_hub_download
        ckpt_file = MODEL_ALIASES[model_name]
        n_layers = 16 if '16' in model_name else 8
        path = hf_hub_download('adamkarvonen/chess_llms', ckpt_file)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        raw_sd = checkpoint['model']
        # Convert nanoGPT -> HF GPT-2 format (transpose Conv1D weights, strip prefix)
        sd = {}
        for k, v in raw_sd.items():
            if k.startswith('_orig_mod.'):
                k = k[len('_orig_mod.'):]
            if '.attn.c_attn.weight' in k or '.attn.c_proj.weight' in k or \
               '.mlp.c_fc.weight' in k or '.mlp.c_proj.weight' in k:
                v = v.T
            sd[k] = v
        from transformers import GPT2Config
        config = GPT2Config(
            vocab_size=32, n_positions=1023, n_embd=512, n_layer=n_layers,
            n_head=8, n_inner=2048, activation_function='gelu',
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel(config)
        model.load_state_dict(sd, strict=False)
        model.to(device)
        tokenizer = None  # Character-level encoding handled by data loader
        model_type = 'gpt2'
        d_model = 512
    elif is_blob_csp:
        # Load dense CSP baseline via native circuit_sparsity loader
        # Workaround: the installed package's GPTConfig doesn't accept all config
        # fields from the blob storage configs, so we filter kwargs manually.
        from circuit_sparsity.inference.gpt import GPTConfig as _CspGPTConfig, GPT as _CspGPT
        import json as _json, inspect as _inspect, io as _io
        from circuit_sparsity.registries import read_file_cached
        blob_base = "https://openaipublic.blob.core.windows.net/circuit-sparsity"
        csp_name = MODEL_ALIASES[model_name]
        model_path = f"{blob_base}/models/{csp_name}"
        beeg_config = _json.loads(read_file_cached(f"{model_path}/beeg_config.json").decode())
        if "n_mlp" in beeg_config:
            beeg_config["d_mlp"] = beeg_config.pop("n_mlp")
        beeg_config["flash"] = False
        beeg_config["grad_checkpointing"] = False
        beeg_config.setdefault("sink", False)
        beeg_config.pop("use_tied_aux_matrix", None)
        # Filter to only accepted kwargs
        valid_keys = set(_inspect.signature(_CspGPTConfig.__init__).parameters.keys()) - {"self"}
        filtered = {k: v for k, v in beeg_config.items() if k in valid_keys}
        config = _CspGPTConfig(**filtered)
        gpt = _CspGPT(config)
        sd = torch.load(
            _io.BytesIO(read_file_cached(f"{model_path}/final_model.pt")),
            weights_only=True, map_location=device
        )
        if "final_logits_bias" not in sd:
            sd["final_logits_bias"] = torch.zeros(config.vocab_size)
        gpt.load_state_dict(sd, strict=False)
        gpt.to(device).eval()
        # Wrap in a namespace so _get_blocks etc. find model.circuit_model.transformer.*
        class _Wrapper(torch.nn.Module):
            def __init__(self, gpt_model):
                super().__init__()
                self.circuit_model = gpt_model
                self.config = type('Config', (), {'vocab_size': gpt_model.config.vocab_size})()
        model = _Wrapper(gpt)
        if device != "cpu":
            model = model.to(device)
        tokenizer = None
        model_type = 'circuitgpt'
        d_model = gpt.config.d_model
    elif is_csp:
        load_dtype = dtype if dtype is not None else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, dtype=load_dtype, trust_remote_code=True
        ).to(device)
        tokenizer = None  # CSP uses random token sequences, no tokenizer needed
        model_type = 'circuitgpt'
        d_model = model.config.d_model
    elif is_gpt2:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model = GPT2LMHeadModel.from_pretrained(hf_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(hf_name)
        model_type = 'gpt2'
        d_model = model.config.n_embd
    elif hf_name in GEMMA3_MULTIMODAL:
        # Multimodal Gemma 3 (4B+): use Gemma3ForCausalLM to extract text decoder
        from transformers import Gemma3ForCausalLM
        load_dtype = dtype if dtype is not None else torch.bfloat16
        model = Gemma3ForCausalLM.from_pretrained(hf_name, torch_dtype=load_dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model_type = 'gemma3'
        d_model = model.config.hidden_size
    else:
        load_dtype = dtype if dtype is not None else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=load_dtype, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        config_type = getattr(model.config, 'model_type', '')
        if config_type.startswith('gemma3'):
            model_type = 'gemma3'
        elif config_type == 'gpt_neo':
            model_type = 'gpt2'  # GPT-Neo has identical structure to GPT-2
        elif config_type == 'gpt_neox':
            model_type = 'llama'  # GPT-NeoX (Pythia) uses rotary embeddings like Llama
        else:
            model_type = 'llama'  # Llama-architecture (Qwen3, Llama, Gemma 1/2, etc.)
        d_model = model.config.hidden_size

    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Attach metadata for dispatch
    model._plateau_info = {'model_type': model_type, 'd_model': d_model}

    return model, tokenizer, device


def get_layer0_output(model, tokenizer, prompt, device):
    """
    Get the Layer 0 output (after block 0) for a prompt.

    This gives us the residual stream state after the first transformer block,
    which we use as the starting point for path optimization.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input text
        device: torch device

    Returns:
        hidden_states: [1, seq_len, hidden_dim] tensor
        input_ids: [1, seq_len] tensor
    """
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = tokens['input_ids']

    # Compute embeddings: token embeddings + position embeddings
    wte = model.transformer.wte(input_ids)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    wpe = model.transformer.wpe(positions)
    hidden_states = wte + wpe

    # Apply dropout and first transformer block
    hidden_states = model.transformer.drop(hidden_states)
    hidden_states = model.transformer.h[0](hidden_states)[0]

    return hidden_states, input_ids


def ablate_attention(model):
    """Register hooks to zero-ablate all attention outputs.

    Zeroes the output of every self-attention module so only MLPs
    and the residual stream contribute to the forward pass.

    Returns list of hook handles (call handle.remove() to restore).
    """
    blocks = _get_blocks(model)
    hooks = []
    for block in blocks:
        attn = getattr(block, 'self_attn', None) or getattr(block, 'attn', None)
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        hooks.append(attn.register_forward_hook(hook_fn))
    return hooks


def ablate_mlp(model):
    """Register hooks to zero-ablate all MLP outputs.

    Zeroes the output of every MLP module so only attention
    and the residual stream contribute to the forward pass.

    Returns list of hook handles (call handle.remove() to restore).
    """
    blocks = _get_blocks(model)
    hooks = []
    for block in blocks:
        mlp = getattr(block, 'mlp', None)
        if mlp is None:
            raise ValueError(f"Cannot find MLP module in block: {type(block)}")
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        hooks.append(mlp.register_forward_hook(hook_fn))
    return hooks


def forward_from_layer1(model, prefix_resid, path_vectors):
    """
    Forward pass from layer 1 onwards, batched over path points.

    This runs the model from layer 1 to the output, allowing us to
    optimize paths in the post-layer-0 residual stream space.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim] - Fixed context (all but last token)
        path_vectors: [n_steps, 1, hidden_dim] - Path points for the last token position

    Returns:
        logits: [n_steps, vocab_size] - Output logits for last token at each path point
    """
    n_steps = path_vectors.shape[0]

    # Expand prefix to batch size
    prefix_batch = prefix_resid.expand(n_steps, -1, -1)

    # Concatenate: [context tokens] + [path point]
    sequence_batch = torch.cat([prefix_batch, path_vectors], dim=1)

    hidden = sequence_batch

    # Pass through layers 1 to n-1
    for block in model.transformer.h[1:]:
        hidden = block(hidden)[0]

    # Final layer norm
    hidden = model.transformer.ln_f(hidden)

    # Project to vocab (last token only)
    last_token_hidden = hidden[:, -1, :]
    logits = model.lm_head(last_token_hidden)

    return logits


def forward_from_layer(model, context, point, start_layer):
    """
    Forward pass from an arbitrary layer, batched over points.

    Generalizes forward_from_layer1 to start from any layer.

    Args:
        model: GPT-2 or Llama-arch model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [n_points, 1, hidden_dim] - Final-token activation(s)
        start_layer: Layer index (0-indexed). Runs blocks [start_layer+1:] onward.

    Returns:
        logits: [n_points, vocab_size] - Output logits for final token
    """
    n_points = point.shape[0]

    context_batch = context.expand(n_points, -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)

    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)

    hidden = _get_final_norm(model)(hidden)
    logits = _get_lm_head(model)(hidden[:, -1, :])

    return logits


def forward_from_layer_to_layer(model, context, point, start_layer, end_layer):
    """
    Forward from start_layer to end_layer (inclusive). Returns last-token residual stream.

    Like forward_from_layer_to_final but stops at a specific intermediate layer
    instead of running through all remaining blocks.

    Args:
        model: GPT-2 or Llama-arch model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [n, 1, hidden_dim] - Last-token activation (may carry gradients)
        start_layer: Layer index (0-indexed). Runs blocks starting from start_layer+1.
        end_layer: Layer index (0-indexed, inclusive). Runs blocks through end_layer.

    Returns:
        final_act: [n, hidden_dim] - Last-token residual stream after end_layer block
    """
    context_batch = context.expand(point.shape[0], -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)

    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:end_layer + 1]:
        hidden = _run_block(block, hidden, model, pos_emb)

    return hidden[:, -1, :]


def forward_from_layer_to_final(model, context, point, start_layer):
    """
    Forward from layer L to final transformer block (no ln_f, no unembedding).

    Returns last-token residual stream after the final transformer block.
    Differentiable w.r.t. point — uses only out-of-place ops.

    Args:
        model: GPT-2 or Llama-arch model
        context: [1, seq_len-1, hidden_dim] - Prefix residual stream at start_layer
        point: [n, 1, hidden_dim] - Last-token activation (may carry gradients)
        start_layer: Layer index (0-indexed). Runs blocks [start_layer+1:] onward.

    Returns:
        final_act: [n, hidden_dim] - Last-token residual stream after final block
    """
    context_batch = context.expand(point.shape[0], -1, -1)
    hidden = torch.cat([context_batch, point], dim=1)

    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)

    return hidden[:, -1, :]


def forward_from_layer_allpos(model, hidden_states, start_layer):
    """
    Forward pass from an arbitrary layer, patching ALL positions.

    Unlike forward_from_layer (which takes separate context + final-token),
    this takes the full residual stream at all positions and runs the
    remaining layers + ln_f + lm_head.

    Args:
        model: GPT-2 or Llama-arch model
        hidden_states: [batch, seq_len, hidden_dim] - Full residual stream at start_layer
        start_layer: Layer index (0-indexed). Runs blocks [start_layer+1:] onward.

    Returns:
        logits: [batch, vocab_size] - Output logits for final token only
    """
    hidden = hidden_states

    blocks = _get_blocks(model)
    pos_emb = None
    if _is_rope_model(model):
        pos_emb = _get_position_embeddings(model, hidden)
    elif _is_cat_pos_emb_model(model):
        pos_emb = _get_cat_pos_emb(model, hidden.shape[1])

    for block in blocks[start_layer + 1:]:
        hidden = _run_block(block, hidden, model, pos_emb)

    hidden = _get_final_norm(model)(hidden)
    logits = _get_lm_head(model)(hidden[:, -1, :])

    return logits


def get_logits_at_point(model, prefix_resid, point, layer=0):
    """
    Get output logits at a single activation point.

    Args:
        model: GPT-2 model
        prefix_resid: [1, seq_len-1, hidden_dim]
        point: [1, 1, hidden_dim]
        layer: Layer index (0-indexed). Default 0 for backward compatibility.

    Returns:
        logits: [vocab_size] tensor
    """
    logits = forward_from_layer(model, prefix_resid, point, layer)
    return logits.squeeze(0)


def extract_activations(model, tokenizer, texts, layer, device, batch_size=1):
    """
    Extract residual stream activations at a given layer for multiple texts.

    For each text, returns the context (prefix positions) and the final-token
    activation at the specified layer. Stores results on CPU to save GPU memory.

    Args:
        model: GPT-2 or Llama-arch model
        tokenizer: tokenizer
        texts: list of strings
        layer: transformer block index (0-indexed)
        device: torch device
        batch_size: texts per batch (1 for variable-length texts)

    Returns:
        contexts: list of [1, seq_len-1, hidden_dim] tensors (CPU)
        activations: [n_texts, hidden_dim] tensor (CPU)
    """
    contexts = []
    activations = []

    for text in texts:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        # hidden_states[layer+1] = residual stream after transformer block `layer`
        hidden = outputs.hidden_states[layer + 1]

        contexts.append(hidden[:, :-1, :].cpu())
        activations.append(hidden[:, -1, :].squeeze(0).cpu())

    activations = torch.stack(activations)

    return contexts, activations


def extract_activations_allpos(model, tokenizer, token_ids_list, layer, device, batch_size=64):
    """
    Extract residual stream activations at ALL positions for fixed-length token sequences.

    Args:
        model: GPT-2 or Llama-arch model
        tokenizer: tokenizer
        token_ids_list: list of [seq_len] tensors (all same length)
        layer: transformer block index (0-indexed)
        device: torch device
        batch_size: sequences per batch

    Returns:
        activations: [n_sequences, seq_len * hidden_dim] tensor (CPU) - concatenated
    """
    all_activations = []

    for i in range(0, len(token_ids_list), batch_size):
        batch_tokens = torch.stack(token_ids_list[i:i+batch_size]).to(device)

        with torch.no_grad():
            outputs = model(batch_tokens, output_hidden_states=True)

        # hidden_states[layer+1] = residual stream after block `layer`
        hidden = outputs.hidden_states[layer + 1]  # [batch, seq_len, d]

        # Flatten positions: [batch, seq_len * d]
        batch_flat = hidden.reshape(hidden.shape[0], -1).cpu()
        all_activations.append(batch_flat)

        if (i // batch_size) % 10 == 0:
            print(f"  Extracted {min(i + batch_size, len(token_ids_list))}/{len(token_ids_list)}", end='\r')

    print()
    return torch.cat(all_activations, dim=0)
