import os
import json
import argparse
import re

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.distillation import (
    DistillationTrainer,
    DistillationConfig,
)


# ============================================================
# args
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--vllm_max_model_length",
    type=int,
    default=None,
    help="Max model length for colocated student vLLM rollout engine.",
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
)

parser.add_argument(
    "--student",
    type=str,
    required=True,
)

parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Path to a checkpoint directory for resuming training.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
)

parser.add_argument(
    "--teacher_server_url",
    type=str,
    default=None,
    help=(
        "URL of the teacher vLLM server. If omitted, --teacher is loaded "
        "locally in the training process."
    ),
)

parser.add_argument(
    "--teacher",
    type=str,
    default=None,
    help="Local teacher model path used when --teacher_server_url is omitted.",
)

parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
)

parser.add_argument(
    "--epochs",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
)

parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=4,
)

parser.add_argument(
    "--generation_batch_size",
    type=int,
    default=4,
)

parser.add_argument(
    "--num_generations",
    type=int,
    default=1,
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-6,
)

parser.add_argument(
    "--beta",
    type=float,
    default=0.0,
    help="Reverse-KL/mixed distillation beta. Use beta > 0 for reverse-style OPD.",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature for student rollouts.",
)

parser.add_argument(
    "--top_p",
    type=float,
    default=0.95,
    help="Nucleus sampling top-p for student rollouts.",
)

parser.add_argument(
    "--max_prompt_length",
    type=int,
    default=4096,
)

parser.add_argument(
    "--max_completion_length",
    type=int,
    default=8192,
)

parser.add_argument(
    "--loss_top_k",
    type=int,
    default=8,
    help=(
        "Teacher top-k probabilities used for forward-KL. "
        "Teacher-server mode requires a positive value."
    ),
)

parser.add_argument(
    "--vllm_gpu_memory_utilization",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--debug_batches",
    type=int,
    default=2,
)

parser.add_argument(
    "--drop_no_boxed_rollout",
    action="store_true",
    help="Skip OPD loss for student rollouts without a boxed final answer.",
)

parser.add_argument(
    "--drop_length_rollout",
    action="store_true",
    help=(
        "Skip OPD loss for rollouts that reach max_completion_length. "
        "This is a practical proxy for finish_reason == length."
    ),
)

parser.add_argument("--save_steps", type=int, default=0)
parser.add_argument("--save_total_limit", type=int, default=3)

parser.add_argument(
    "--qc_log_every",
    type=int,
    default=10,
    help="Print quality-control rollout statistics every N generation slices.",
)

parser.add_argument(
    "--prompt_style",
    choices=["chat", "completion"],
    default="chat",
    help=(
        "Use chat messages for Instruct models, or plain completion prompts "
        "for Base models."
    ),
)

parser.add_argument(
    "--instruction",
    type=str,
    default=(
        "Please reason step by step, "
        "and put your final answer within \\boxed{}."
    ),
    help="Instruction appended to each problem before on-policy rollout.",
)

args_cli = parser.parse_args()


# ============================================================
# basic checks
# ============================================================

if args_cli.batch_size < 1:
    raise ValueError("--batch_size must be >= 1")

if args_cli.gradient_accumulation_steps < 1:
    raise ValueError(
        "--gradient_accumulation_steps must be >= 1"
    )

if args_cli.generation_batch_size < 1:
    raise ValueError(
        "--generation_batch_size must be >= 1"
    )

if args_cli.num_generations < 1:
    raise ValueError("--num_generations must be >= 1")

if args_cli.temperature <= 0:
    raise ValueError("--temperature must be > 0")

if not (0 < args_cli.top_p <= 1):
    raise ValueError("--top_p must be in (0, 1]")

if args_cli.max_prompt_length < 1:
    raise ValueError("--max_prompt_length must be >= 1")

if args_cli.max_completion_length < 1:
    raise ValueError(
        "--max_completion_length must be >= 1"
    )

if args_cli.loss_top_k < 1:
    raise ValueError(
        "--loss_top_k must be >= 1"
    )

if args_cli.teacher_server_url is None and args_cli.teacher is None:
    raise ValueError(
        "Provide either --teacher_server_url for server teacher mode "
        "or --teacher for local teacher mode."
    )

if args_cli.teacher_server_url is not None and args_cli.teacher is not None:
    raise ValueError(
        "Use only one teacher source: --teacher_server_url or --teacher."
    )

if not 0.0 < args_cli.vllm_gpu_memory_utilization < 1.0:
    raise ValueError(
        "--vllm_gpu_memory_utilization must be between 0 and 1"
    )

if args_cli.qc_log_every < 1:
    raise ValueError("--qc_log_every must be >= 1")


expected_generation_batch = (
    args_cli.batch_size
    * args_cli.gradient_accumulation_steps
)

actual_generation_batch = (
    args_cli.generation_batch_size
    * args_cli.num_generations
)

if actual_generation_batch != expected_generation_batch:
    raise ValueError(
        "TRL requires:\n"
        "generation_batch_size * num_generations "
        "== batch_size * gradient_accumulation_steps\n"
        f"Got {args_cli.generation_batch_size} "
        f"* {args_cli.num_generations} "
        f"!= {args_cli.batch_size} "
        f"* {args_cli.gradient_accumulation_steps}"
    )


# ============================================================
# load dataset
# ============================================================

with open(args_cli.input, "r", encoding="utf-8") as f:
    raw = json.load(f)

if not isinstance(raw, list):
    raise ValueError(
        "Input JSON must contain a list of samples."
    )


instruction = args_cli.instruction

rows = []

for index, sample in enumerate(raw):
    if "problem" not in sample:
        raise KeyError(
            f"Sample {index} does not contain "
            "the required 'problem' field."
        )

    problem = str(sample["problem"]).strip()

    if not problem:
        continue

    prompt = problem + "\n\n" + instruction + "\n\n"

    if args_cli.prompt_style == "completion":
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
    else:
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            problem
                            + "\n\n"
                            + instruction
                        ),
                    }
                ]
            }
        )

if not rows:
    raise ValueError(
        "No valid samples were found in the input file."
    )

dataset = Dataset.from_list(rows)

print("=" * 80)
print("DATASET")
print(dataset)
print(dataset[0])
print("=" * 80)


# ============================================================
# tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    args_cli.student,
    trust_remote_code=True,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ============================================================
# disable Qwen3 thinking for chat-style Instruct runs only
# ============================================================

if args_cli.prompt_style == "completion":
    def apply_completion_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        if isinstance(messages, list):
            text = "\n".join(
                str(message.get("content", ""))
                for message in messages
            )
        else:
            text = str(messages)

        if tokenize:
            return tokenizer(
                text,
                add_special_tokens=True,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key in {
                        "padding",
                        "truncation",
                        "max_length",
                        "return_tensors",
                    }
                },
            )["input_ids"]

        return text

    tokenizer.apply_chat_template = apply_completion_template

elif args_cli.prompt_style == "chat":
    _original_apply_chat_template = (
        tokenizer.apply_chat_template
    )

    def apply_chat_template_no_thinking(
        *args,
        **kwargs,
    ):
        try:
            kwargs["enable_thinking"] = False
            return _original_apply_chat_template(
                *args,
                **kwargs,
            )
        except TypeError as error:
            if "enable_thinking" not in str(error):
                raise
            kwargs.pop("enable_thinking", None)

        return _original_apply_chat_template(
            *args,
            **kwargs,
        )

    tokenizer.apply_chat_template = (
        apply_chat_template_no_thinking
    )


# ============================================================
# config
# ============================================================

total_max_length = (
    args_cli.max_prompt_length
    + args_cli.max_completion_length
)

config = DistillationConfig(
    output_dir=args_cli.output,
    ignore_data_skip=True,
    vllm_max_model_length=args_cli.vllm_max_model_length,
    # --------------------------------------------------------
    # training
    # --------------------------------------------------------
    num_train_epochs=args_cli.epochs,
    max_steps=args_cli.max_steps,

    per_device_train_batch_size=args_cli.batch_size,

    gradient_accumulation_steps=(
        args_cli.gradient_accumulation_steps
    ),

    generation_batch_size=(
        args_cli.generation_batch_size
    ),

    num_generations=args_cli.num_generations,
    learning_rate=args_cli.learning_rate,

    bf16=True,
    gradient_checkpointing=True,

    logging_steps=1,

    save_strategy="steps" if args_cli.save_steps > 0 else "no",
    save_steps=args_cli.save_steps if args_cli.save_steps > 0 else None,
    save_total_limit=args_cli.save_total_limit,

    report_to="none",

    # 后面保持不变
    lmbda=1.0,
    beta=args_cli.beta,

    loss_top_k=args_cli.loss_top_k,
    loss_add_tail=True,
    temperature=args_cli.temperature,
    top_p=args_cli.top_p,

    use_liger_kernel=False,

    max_prompt_length=args_cli.max_prompt_length,
    max_completion_length=args_cli.max_completion_length,
    max_length=total_max_length,

    use_teacher_server=args_cli.teacher_server_url is not None,
    teacher_model_server_url=args_cli.teacher_server_url,
    teacher_model_name_or_path=args_cli.teacher,
    teacher_model_init_kwargs=(
        {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if args_cli.teacher is not None
        else None
    ),

    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=(
        args_cli.vllm_gpu_memory_utilization
    ),
    vllm_tensor_parallel_size=1,
)


print("=" * 80)
print("DISTILLATION CONFIG")
print(config)
print("=" * 80)

print("Distillation setup:")
print("  rollout              : student")
print(
    "  teacher              : "
    + (
        "external server"
        if args_cli.teacher_server_url is not None
        else "local model"
    )
)
print(
    "  objective            : "
    + ("reverse/mixed KL" if args_cli.beta > 0 else "forward KL")
)
print(f"  teacher server       : {args_cli.teacher_server_url}")
print(f"  local teacher        : {args_cli.teacher}")
print(f"  beta                 : {args_cli.beta}")
print(f"  loss_top_k           : {args_cli.loss_top_k}")
print(f"  temperature          : {args_cli.temperature}")
print(f"  top_p                : {args_cli.top_p}")
print(f"  instruction          : {args_cli.instruction}")
print(f"  max prompt length    : {args_cli.max_prompt_length}")
print(f"  max completion length: {args_cli.max_completion_length}")
print(
    "  generation batch    : "
    f"{args_cli.generation_batch_size}"
)
print(
    "  effective train batch: "
    f"{expected_generation_batch}"
)
print(
    "  student vLLM memory : "
    f"{args_cli.vllm_gpu_memory_utilization}"
)
print("  QC drop no boxed    :", args_cli.drop_no_boxed_rollout)
print("  QC drop length      :", args_cli.drop_length_rollout)


# ============================================================
# quality-controlled trainer
# ============================================================

BOXED_RE = re.compile(r"\\boxed\s*\{[^{}]*\}")


class QualityControlledDistillationTrainer(DistillationTrainer):
    def __init__(
        self,
        *args,
        drop_no_boxed_rollout=False,
        drop_length_rollout=False,
        qc_log_every=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drop_no_boxed_rollout = drop_no_boxed_rollout
        self.drop_length_rollout = drop_length_rollout
        self.qc_log_every = qc_log_every
        self.qc_calls = 0
        self.qc_stats = {
            "generated": 0,
            "kept": 0,
            "dropped": 0,
            "dropped_no_boxed": 0,
            "dropped_length": 0,
        }

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        if labels is not None:
            keep_mask = (labels != -100).any(dim=1)
            if not bool(keep_mask.all().item()):
                kept = int(keep_mask.sum().item())
                total = int(keep_mask.numel())
                if kept == 0:
                    zero = next(model.parameters()).sum() * 0.0
                    return (zero, None) if return_outputs else zero

                filtered_inputs = {}
                for key, value in inputs.items():
                    if (
                        torch.is_tensor(value)
                        and value.shape[:1] == keep_mask.shape
                    ):
                        filtered_inputs[key] = value[keep_mask]
                    else:
                        filtered_inputs[key] = value

                print(
                    "[QC] loss batch filtered: "
                    f"kept={kept} dropped={total - kept}"
                )
                inputs = filtered_inputs

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _store_completions_in_buffer(
        self,
        slices,
        on_policy_indices,
        local_slice_indices,
        local_prompts,
        completion_ids,
    ):
        device = self.accelerator.device
        pad_token_id = (
            self.processing_class.pad_token_id
            if self.processing_class.pad_token_id is not None
            else 0
        )
        max_completion_length = self.generation_config.max_new_tokens

        slice_completions = {idx: [] for idx in on_policy_indices}
        slice_prompt_ids = {idx: [] for idx in on_policy_indices}
        for i, slice_idx in enumerate(local_slice_indices):
            slice_completions[slice_idx].append(completion_ids[i])
            slice_prompt_ids[slice_idx].append(local_prompts[i])

        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            prompt_id_tensors = slice_prompt_ids[slice_idx]
            prompt_width = max(len(p) for p in prompt_id_tensors)
            prompt_token_lengths = torch.tensor(
                [len(p) for p in prompt_id_tensors],
                device=device,
                dtype=torch.long,
            )
            prompt_attention_mask = (
                torch.arange(prompt_width, device=device).unsqueeze(0)
                >= (prompt_width - prompt_token_lengths).unsqueeze(1)
            ).long()

            prompt_ids = torch.stack(
                [
                    F.pad(
                        p,
                        (prompt_width - len(p), 0),
                        value=pad_token_id,
                    )
                    for p in prompt_id_tensors
                ]
            ).to(device)

            completion_tensors = []
            completion_ids_for_text = []
            completion_lengths = []
            for comp_ids in slice_completions[slice_idx]:
                t = torch.tensor(comp_ids, device=device)
                if len(t) > max_completion_length:
                    t = t[:max_completion_length]
                completion_ids_for_text.append(t.tolist())
                completion_lengths.append(len(t))
                if len(t) < max_completion_length:
                    padding = torch.full(
                        (max_completion_length - len(t),),
                        pad_token_id,
                        device=device,
                        dtype=t.dtype,
                    )
                    t = torch.cat([t, padding])
                completion_tensors.append(t)

            completion_ids_padded = torch.stack(completion_tensors)
            new_input_ids = torch.cat(
                [prompt_ids, completion_ids_padded],
                dim=1,
            )
            completion_lengths = torch.tensor(
                completion_lengths,
                device=device,
                dtype=torch.long,
            )
            new_attention_mask, new_labels = self._build_sequence_batch(
                new_input_ids,
                prompt_width,
                prompt_token_lengths,
                completion_lengths,
            )

            prompt_texts = self.processing_class.batch_decode(
                prompt_id_tensors,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            completion_texts = self.processing_class.batch_decode(
                completion_ids_for_text,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            bad_no_boxed = torch.zeros(
                len(completion_texts),
                device=device,
                dtype=torch.bool,
            )
            bad_length = torch.zeros_like(bad_no_boxed)

            if self.drop_no_boxed_rollout:
                bad_no_boxed = torch.tensor(
                    [
                        BOXED_RE.search(text) is None
                        for text in completion_texts
                    ],
                    device=device,
                    dtype=torch.bool,
                )

            if self.drop_length_rollout:
                bad_length = completion_lengths >= max_completion_length

            bad_mask = bad_no_boxed | bad_length
            if bad_mask.any():
                completion_label_mask = new_labels != -100
                new_labels[
                    bad_mask.unsqueeze(1).expand_as(completion_label_mask)
                    & completion_label_mask
                ] = -100

            generated = int(bad_mask.numel())
            dropped_no_boxed = int(bad_no_boxed.sum().item())
            dropped_length = int(bad_length.sum().item())
            dropped = int(bad_mask.sum().item())
            kept = generated - dropped
            self.qc_stats["generated"] += generated
            self.qc_stats["kept"] += kept
            self.qc_stats["dropped"] += dropped
            self.qc_stats["dropped_no_boxed"] += dropped_no_boxed
            self.qc_stats["dropped_length"] += dropped_length
            self.qc_calls += 1

            if self.qc_calls % self.qc_log_every == 0:
                total = max(self.qc_stats["generated"], 1)
                print(
                    "[QC] "
                    f"generated={self.qc_stats['generated']} "
                    f"kept={self.qc_stats['kept']} "
                    f"dropped={self.qc_stats['dropped']} "
                    f"drop_rate={self.qc_stats['dropped'] / total:.2%} "
                    f"no_boxed={self.qc_stats['dropped_no_boxed']} "
                    f"length={self.qc_stats['dropped_length']}"
                )

            updated = dict(slice_inputs)
            updated["input_ids"] = new_input_ids
            updated["attention_mask"] = new_attention_mask
            updated["labels"] = new_labels
            updated["prompts"] = prompt_ids
            updated["prompt_attention_mask"] = prompt_attention_mask

            self._buffered_inputs[slice_idx] = updated
            self._buffered_text_logs[slice_idx] = (
                prompt_texts,
                completion_texts,
            )


# ============================================================
# trainer
# ============================================================

trainer_kwargs = {}
if args_cli.teacher is not None:
    trainer_kwargs["teacher_model"] = args_cli.teacher

trainer = QualityControlledDistillationTrainer(
    model=args_cli.student,

    args=config,

    train_dataset=dataset,
    processing_class=tokenizer,
    drop_no_boxed_rollout=args_cli.drop_no_boxed_rollout,
    drop_length_rollout=args_cli.drop_length_rollout,
    qc_log_every=args_cli.qc_log_every,
    **trainer_kwargs,
)


# ============================================================
# debug rollout
# ============================================================

counter = {"n": 0}

vllm_generation = getattr(
    trainer,
    "vllm_generation",
    None,
)

if (
    vllm_generation is not None
    and hasattr(vllm_generation, "llm")
    and hasattr(vllm_generation.llm, "generate")
):
    old_generate = (
        vllm_generation.llm.generate
    )

    def debug_generate(*args, **kwargs):
        outputs = old_generate(*args, **kwargs)

        if counter["n"] < args_cli.debug_batches:
            print("\n" + "=" * 100)
            print("ROLLOUT DEBUG")
            print("=" * 100)

            for i, output in enumerate(outputs):
                if not output.outputs:
                    print(
                        f"\n--- sample {i}: "
                        "no output ---"
                    )
                    continue

                generated = output.outputs[0]

                text = generated.text
                token_ids = generated.token_ids

                print(f"\n--- sample {i} ---")
                print(text[:2000])

                print(
                    "thinking:",
                    "<think>" in text,
                )

                print(
                    "tokens:",
                    len(token_ids),
                )

                finish_reason = getattr(
                    generated,
                    "finish_reason",
                    None,
                )

                print(
                    "finish_reason:",
                    finish_reason,
                )

            counter["n"] += 1

        return outputs

    vllm_generation.llm.generate = debug_generate

else:
    print(
        "Warning: student vLLM generate method "
        "was not found. Rollout debug is disabled."
    )


# ============================================================
# train
# ============================================================

print("=" * 80)
print("START TRAINING")
print("=" * 80)

train_result = trainer.train(
    resume_from_checkpoint=(
        args_cli.resume_from_checkpoint
    )
)

print("=" * 80)
print("TRAINING FINISHED")
print(train_result)
print("=" * 80)


# ============================================================
# save final
# ============================================================

final_path = os.path.join(
    args_cli.output,
    "final",
)

trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print("=" * 80)
print("MODEL SAVED")
print(final_path)
print("=" * 80)
