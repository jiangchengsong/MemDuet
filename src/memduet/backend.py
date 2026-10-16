"""Native HF causal backend with immutable token traces and fixed-reference KL.

Gate token validation follows MemAgent/gate_sft/prompting.py (see NOTICE).
Generation probabilities are raw model probabilities before sampling warpers.
Training rollouts must therefore use temperature=top_p=1; inference may differ.
Streaming projection is restricted to explicitly supported native architectures
whose final logits are the untransformed output head. Other models fail clearly.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

from .prompting import apply_chat_template, validate_token_protocol
from .types import GateScore, GenerationEvent, TokenStatistics


class HFBackend:
    # These native causal forwards apply the output head without a subsequent
    # logit scaling/softcapping transform. GPT-2 supports offline CPU tests.
    SUPPORTED_MODEL_TYPES = frozenset(
        {
            "gpt2",
            "qwen2",
            "qwen2_moe",
            "qwen3",
            "qwen3_moe",
            "qwen3_next",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        }
    )

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_context_tokens: int | None = None,
        logits_chunk_size: int = 128,
        score_batch_size: int = 4,
        version: str | None = None,
        provenance: dict | None = None,
        reference_backend=None,
        save_backbone: bool = False,
    ):
        if (
            isinstance(logits_chunk_size, bool)
            or not isinstance(logits_chunk_size, int)
            or logits_chunk_size <= 0
        ):
            raise ValueError("logits_chunk_size must be a positive integer")
        if (
            isinstance(score_batch_size, bool)
            or not isinstance(score_batch_size, int)
            or score_batch_size <= 0
        ):
            raise ValueError("score_batch_size must be a positive integer")
        self.score_batch_size = score_batch_size
        self.model, self.tokenizer = model, tokenizer
        self.reference_backend = reference_backend
        self.save_backbone = save_backbone
        self.protocol = validate_token_protocol(tokenizer)
        self.logits_chunk_size = logits_chunk_size
        self.model.eval()
        config = (
            model.config.get_text_config()
            if hasattr(model.config, "get_text_config")
            else model.config
        )
        model_type = getattr(config, "model_type", None)
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"unsupported causal architecture {model_type!r}: faithful final-logit projection is only validated for {sorted(self.SUPPORTED_MODEL_TYPES)}"
            )
        limits = [
            getattr(config, name, None)
            for name in ("max_position_embeddings", "n_positions")
        ]
        limits.append(getattr(tokenizer, "model_max_length", None))
        limits = [v for v in limits if isinstance(v, int) and 0 < v < 10**9]
        if max_context_tokens is not None:
            if (
                isinstance(max_context_tokens, bool)
                or not isinstance(max_context_tokens, int)
                or max_context_tokens <= 0
            ):
                raise ValueError("max_context_tokens must be a positive integer")
            limits.append(max_context_tokens)
        if not limits:
            raise ValueError(
                "model context capacity is unknown; supply max_context_tokens"
            )
        self.max_context_tokens = min(limits)
        # Generation configuration is authoritative, including its complete
        # multi-EOS list. Fall back only when metadata is absent; tokenizer
        # EOS must not silently narrow a model's configured stopping set.
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(config, "eos_token_id", None)
        if eos is None:
            eos = getattr(tokenizer, "eos_token_id", None)
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        vocabulary_size = self.model.get_output_embeddings().weight.shape[0]
        if not eos_ids or any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < vocabulary_size
            for i in eos_ids
        ):
            raise ValueError(
                "model/tokenizer must define valid EOS token IDs within the output vocabulary"
            )
        self.eos_token_ids = set(eos_ids)
        self._provenance = dict(provenance or {})
        self._provenance.update(
            k_id=self.protocol.k_id,
            u_id=self.protocol.u_id,
            eos_token_ids=sorted(self.eos_token_ids),
            max_context_tokens=self.max_context_tokens,
            score_batch_size=self.score_batch_size,
            model_type=getattr(config, "model_type", type(model).__name__),
        )
        causal = (
            self.model.get_base_model()
            if hasattr(self.model, "peft_config")
            else self.model
        )
        self._last_logits_supported = (
            "logits_to_keep" in inspect.signature(causal.forward).parameters
        )
        self.version = version or json.dumps(self._provenance, sort_keys=True)

    @classmethod
    def from_config(cls, spec: dict, trainable: bool = False):
        # Imports stay lazy: package imports and CLI help never load models.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = dict(spec)
        # Stage III snapshots carry their trained Stage II backbone. Prefer it
        # over the initializer path, which may no longer exist after transfer.
        if spec.get("adapter_path"):
            backbone = Path(spec["adapter_path"]) / "backbone"
            if (backbone / "config.json").is_file():
                spec.update(base_model=str(backbone))
        mode = spec.get("training_mode", "lora")
        if mode not in {"lora", "full"}:
            raise ValueError("training_mode must be lora or full")
        if mode == "full" and spec.get("adapter_path"):
            raise ValueError("full-parameter training cannot load an adapter_path")
        if not isinstance(spec.get("base_model"), str) or not spec["base_model"]:
            raise ValueError("base_model is required")
        dtype_name = spec.get("dtype", "float32")
        if dtype_name not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        device = spec.get("device", "auto")
        if (
            device != "auto"
            and device != "cpu"
            and not (
                isinstance(device, str)
                and device.startswith("cuda:")
                and device[5:].isdigit()
            )
        ):
            raise ValueError("device must be auto, cpu, or cuda:N")
        tokenizer_source = spec["base_model"]
        adapter_path = spec.get("adapter_path")
        if adapter_path and (Path(adapter_path) / "tokenizer_config.json").is_file():
            tokenizer_source = adapter_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=False,
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs an EOS token to initialize padding")
            tokenizer.pad_token = tokenizer.eos_token
        kwargs = dict(dtype=getattr(torch, dtype_name), trust_remote_code=False)
        device_map = spec.get("device_map")
        if device_map is not None:
            if (
                not isinstance(device_map, dict)
                or not device_map
                or any(
                    isinstance(value, bool)
                    or not (
                        isinstance(value, int)
                        and value >= 0
                        or isinstance(value, str)
                        and value.startswith("cuda:")
                        and value[5:].isdigit()
                    )
                    for value in device_map.values()
                )
            ):
                raise ValueError(
                    "explicit device_map must map module names to CUDA devices"
                )
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                raise ValueError(
                    "explicit device_map is single-process only; use one-device replicas with torchrun"
                )
            kwargs["device_map"] = device_map
        elif device == "auto":
            kwargs["device_map"] = "auto"
        if spec.get("attn_implementation"):
            kwargs["attn_implementation"] = spec["attn_implementation"]
        # Transformers >=5 supports extracting Qwen3.5's text config and
        # checkpoint prefix conversion for full multimodal source checkpoints.
        model = AutoModelForCausalLM.from_pretrained(spec["base_model"], **kwargs)
        if device != "auto" and device_map is None:
            model.to(torch.device(device))
        model.requires_grad_(False)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, adapter_path, is_trainable=trainable
            )
        elif trainable and mode == "lora":
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=spec.get("lora_rank", 32),
                    lora_alpha=spec.get("lora_alpha", 32),
                    lora_dropout=spec.get("lora_dropout", 0.0),
                    target_modules=spec.get("lora_targets", "all-linear"),
                    task_type="CAUSAL_LM",
                ),
            )
        elif trainable:
            model.requires_grad_(True)
        if not trainable:
            model.requires_grad_(False)
        if spec.get("gradient_checkpointing") and trainable:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        provenance = {
            "base_model": spec["base_model"],
            "dtype": dtype_name,
            "adapter_path": adapter_path,
            "training_mode": mode,
        }
        reference_backend = None
        if trainable and spec.get("reference_model"):
            reference_spec = dict(spec["reference_model"])
            if reference_spec.get("adapter_path") or reference_spec.get(
                "reference_model"
            ):
                raise ValueError(
                    "reference_model must be a standalone frozen initial model"
                )
            reference_spec.setdefault("dtype", dtype_name)
            reference_spec.setdefault("device", device)
            reference_backend = cls.from_config(reference_spec, trainable=False)
            if (
                tokenizer.get_vocab() != reference_backend.tokenizer.get_vocab()
                or tokenizer.chat_template != reference_backend.tokenizer.chat_template
                or model.get_output_embeddings().weight.shape[0]
                != reference_backend.model.get_output_embeddings().weight.shape[0]
            ):
                raise ValueError(
                    "generation reference and policy token vocabularies/templates must match"
                )
            provenance["reference"] = reference_backend.provenance()
            saved_path = (
                Path(adapter_path or spec["base_model"]) / "memduet_backend.json"
            )
            if saved_path.is_file():
                recorded = json.loads(saved_path.read_text()).get("reference", {})
                expected = recorded.get("base_model")
                if expected is not None and expected != provenance["reference"].get(
                    "base_model"
                ):
                    raise ValueError(
                        "fixed generation reference model differs from saved checkpoint"
                    )
        return cls(
            model,
            tokenizer,
            max_context_tokens=spec.get("max_context_tokens"),
            score_batch_size=spec.get("score_batch_size", 4),
            provenance=provenance,
            reference_backend=reference_backend,
            save_backbone=spec.get("save_backbone", False),
        )

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    def provenance(self):
        return dict(self._provenance, version=self.version)

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if hasattr(self.model, "peft_config"):
            self._provenance["adapter_path"] = str(path)
            if self.save_backbone:
                if any(
                    p.requires_grad
                    for n, p in self.model.named_parameters()
                    if "lora_" not in n
                ):
                    raise ValueError(
                        "portable LoRA checkpoint requires a frozen backbone"
                    )
                causal = self.model.get_base_model()
                # Save the unmerged backbone: merging here would apply Stage
                # III deltas twice when its separately saved adapter reloads.
                state = {
                    name.replace(".base_layer.", "."): value.detach()
                    for name, value in causal.state_dict().items()
                    if ".lora_" not in name
                }
                causal.save_pretrained(path / "backbone", state_dict=state)
                self.tokenizer.save_pretrained(path / "backbone")
                self._provenance["base_model"] = str(path / "backbone")
        else:
            self._provenance["base_model"] = str(path)
        (path / "memduet_backend.json").write_text(
            json.dumps(self.provenance(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def encode_prompt(self, prompt: str) -> list[int]:
        ids = apply_chat_template(self.tokenizer, prompt)
        self._check_context(ids)
        return ids

    def _check_context(self, ids, reserve=0):
        if not ids or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids
        ):
            raise ValueError("input token IDs must be nonempty nonnegative integers")
        if len(ids) + reserve > self.max_context_tokens:
            raise ValueError(
                f"model context overflow: {len(ids)} input + {reserve} reserved > {self.max_context_tokens}"
            )

    def _inputs(self, ids):
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        return dict(input_ids=tokens, attention_mask=torch.ones_like(tokens))

    def score_tensor(self, prompt_ids):
        ids = (
            prompt_ids.tolist()
            if isinstance(prompt_ids, torch.Tensor)
            else list(prompt_ids)
        )
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("score_tensor expects one prompt")
            ids = ids[0]
        self._check_context(ids)
        # Decoder hidden states avoid allocating [prompt length, vocabulary].
        hidden = self._hidden(ids)[0, -1:]
        logits = self._project_logits(hidden)[
            0, [self.protocol.k_id, self.protocol.u_id]
        ]
        return logits.log_softmax(-1)

    def _project_logits(self, hidden):
        # Constructor validates native architecture semantics before callers
        # may use this streaming equivalent of the full causal forward.
        return self.model.get_output_embeddings()(hidden).float()

    @torch.no_grad()
    def score(self, prompt):
        self.model.eval()
        ids = self.encode_prompt(prompt)
        value = self.score_tensor(ids)[1].exp().item()
        return GateScore(prompt, ids, value)

    @torch.no_grad()
    def score_many(self, prompts):
        self.model.eval()
        prompts = list(prompts)
        encoded = [self.encode_prompt(prompt) for prompt in prompts]
        scores = []
        for start in range(0, len(prompts), self.score_batch_size):
            batch = encoded[start : start + self.score_batch_size]
            width = max(map(len, batch))
            tokens = torch.full(
                (len(batch), width),
                self.protocol.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            mask = torch.zeros_like(tokens)
            for row, ids in enumerate(batch):
                tokens[row, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=tokens.device
                )
                mask[row, : len(ids)] = 1
            try:
                hidden = self._hidden_inputs(tokens, mask)
            except RuntimeError as exc:
                raise RuntimeError(
                    "batched gate decoder failed; set score_batch_size=1 to disable padded batching for this model/kernel"
                ) from exc
            rows = torch.arange(len(batch), device=hidden.device)
            last = torch.tensor([len(ids) - 1 for ids in batch], device=hidden.device)
            # The assistant-first distribution lives at each prompt's final
            # real token, not at the final padded column.
            logits = self._project_logits(hidden[rows, last])[
                :, [self.protocol.k_id, self.protocol.u_id]
            ]
            probabilities = logits.log_softmax(-1)[:, 1].exp().tolist()
            scores.extend(
                GateScore(prompt, ids, probability)
                for prompt, ids, probability in zip(
                    prompts[start : start + self.score_batch_size], batch, probabilities
                )
            )
        return scores

    @torch.no_grad()
    def generate(self, prompt, *, kind, max_new_tokens, temperature, top_p):
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            not isinstance(temperature, (int, float))
            or not torch.isfinite(torch.tensor(temperature))
            or temperature < 0
        ):
            raise ValueError("temperature must be finite and nonnegative")
        if (
            not isinstance(top_p, (int, float))
            or not torch.isfinite(torch.tensor(top_p))
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be finite and in (0, 1]")
        if kind not in {"bootstrap", "memory", "answer"}:
            raise ValueError("unknown generation kind")
        self.model.eval()
        prompt_ids = self.encode_prompt(prompt)
        self._check_context(prompt_ids, max_new_tokens)
        token_ids, log_probs = [], []
        past = None
        for _ in range(max_new_tokens):
            # Model cache implementations differ. Standard generation input
            # preparation supplies cache_position/position_ids when required.
            full_ids = prompt_ids + token_ids
            inputs = self._inputs(full_ids)
            if past is not None:
                inputs["input_ids"] = inputs["input_ids"][:, -1:]
                prepared = self.model.prepare_inputs_for_generation(
                    **inputs, past_key_values=past, use_cache=True
                )
            else:
                prepared = dict(inputs, use_cache=True)
            if self._last_logits_supported:
                prepared["logits_to_keep"] = 1
            output = self.model(**prepared)
            logits = output.logits[0, -1].float()
            raw = logits.log_softmax(-1)
            if not torch.isfinite(raw).all():
                raise ValueError("model generated nonfinite token log probabilities")
            if temperature == 0:
                token = int(logits.argmax())
            else:
                sampling_logits = logits / temperature
                if top_p < 1:
                    sorted_logits, indices = sampling_logits.sort(descending=True)
                    remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    sampling_logits = sampling_logits.clone()
                    sampling_logits[indices[remove]] = -torch.inf
                token = int(torch.multinomial(sampling_logits.softmax(-1), 1))
            token_ids.append(token)
            log_probs.append(float(raw[token]))
            past = output.past_key_values
            if token in self.eos_token_ids:
                break
        text = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GenerationEvent(kind, prompt, prompt_ids, token_ids, log_probs, text)

    def _hidden(self, ids):
        inputs = self._inputs(ids)
        return self._hidden_inputs(inputs["input_ids"], inputs["attention_mask"])

    def _hidden_inputs(self, input_ids, attention_mask):
        causal = (
            self.model.get_base_model()
            if hasattr(self.model, "peft_config")
            else self.model
        )
        decoder = causal.base_model
        if decoder is causal:
            raise ValueError(
                "model must expose a decoder base_model for streaming replay"
            )

        def decode(tokens, mask):
            result = decoder(
                input_ids=tokens, attention_mask=mask, use_cache=False, return_dict=True
            )
            if not hasattr(result, "last_hidden_state"):
                raise ValueError("causal decoder does not expose last_hidden_state")
            return result.last_hidden_state

        # HF's built-in layer checkpointing commonly activates only in train
        # mode. RL requires eval mode for stable dropout-free replay, so use
        # an explicit decoder checkpoint when that option has been enabled.
        if (
            torch.is_grad_enabled()
            and not self.model.training
            and getattr(causal, "is_gradient_checkpointing", False)
            and any(p.requires_grad for p in decoder.parameters())
        ):
            return checkpoint(decode, input_ids, attention_mask, use_reentrant=False)
        return decode(input_ids, attention_mask)

    def token_statistics(self, event: GenerationEvent, *, reference: bool = True):
        self.model.eval()
        self._check_context(event.prompt_ids)
        self._check_context(event.prompt_ids + event.token_ids)
        if not event.token_ids:
            empty = torch.empty(0, device=self.device)
            return TokenStatistics(empty, empty)
        adapter = hasattr(self.model, "disable_adapter") and hasattr(
            self.model, "peft_config"
        )
        external = self.reference_backend
        if (
            reference
            and external is None
            and not adapter
            and any(p.requires_grad for p in self.model.parameters())
        ):
            raise ValueError(
                "trainable replay requires a fixed reference_model or an adapter-disabled frozen base"
            )
        if (
            reference
            and external is None
            and adapter
            and any(
                p.requires_grad
                for name, p in self.model.named_parameters()
                if "lora_" not in name
            )
        ):
            raise ValueError(
                "adapter-disabled reference requires frozen base parameters"
            )
        replay_ids = event.prompt_ids + event.token_ids[:-1]
        first = len(event.prompt_ids) - 1
        hidden = self._hidden(replay_ids)[0, first:]
        reference_hidden = None
        if reference:
            baseline_backend = external or self
            baseline_backend._check_context(replay_ids)
            with (
                torch.no_grad(),
                self.model.disable_adapter()
                if adapter and external is None
                else nullcontext(),
            ):
                baseline_backend.model.eval()
                reference_hidden = baseline_backend._hidden(replay_ids)[0, first:]
        targets = torch.tensor(event.token_ids, device=hidden.device)
        head = self.model.get_output_embeddings()
        probabilities, divergences = [], []
        for start in range(0, len(targets), self.logits_chunk_size):
            end = min(start + self.logits_chunk_size, len(targets))
            selected = targets[start:end]

            # Bind chunk-local values for checkpoint's delayed backward replay.
            def project(current, baseline, selected=selected):
                logits = self._project_logits(current).log_softmax(-1)
                token_log_probs = logits.gather(
                    -1, selected.to(logits.device).unsqueeze(-1)
                ).squeeze(-1)
                if reference:
                    with (
                        torch.no_grad(),
                        self.model.disable_adapter()
                        if adapter and external is None
                        else nullcontext(),
                    ):
                        base = (
                            baseline_backend._project_logits(baseline)
                            .log_softmax(-1)
                            .to(logits.device)
                        )
                    divergence = (logits.exp() * (logits - base)).sum(-1)
                else:
                    divergence = torch.zeros_like(token_log_probs)
                return token_log_probs, divergence

            baseline = (
                reference_hidden[start:end] if reference else hidden[start:end].detach()
            )
            if torch.is_grad_enabled() and (
                hidden.requires_grad or any(p.requires_grad for p in head.parameters())
            ):
                lp, kl = checkpoint(
                    project, hidden[start:end], baseline, use_reentrant=False
                )
            else:
                lp, kl = project(hidden[start:end], baseline)
            probabilities.append(lp)
            divergences.append(kl)
        return TokenStatistics(torch.cat(probabilities), torch.cat(divergences))
