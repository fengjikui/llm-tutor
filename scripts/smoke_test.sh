#!/usr/bin/env bash
set -euo pipefail

uv run pytest -q
uv run python -m llm_tutor.experiments.manual_gradient_descent
uv run python -m llm_tutor.experiments.train_linear_classifier --epochs 3
uv run python -m llm_tutor.experiments.train_neural_network_basics --epochs 3
uv run python -m llm_tutor.experiments.compare_training_strategies --epochs 2
uv run python -m llm_tutor.experiments.train_cnn --epochs 1 --train-limit 512 --val-limit 128 --test-limit 128
uv run python -m llm_tutor.experiments.train_rnn_classifier --epochs 2
uv run python -m llm_tutor.experiments.compare_recurrent_cells --epochs 1
uv run python -m llm_tutor.experiments.train_seq2seq_translation --epochs 2
uv run python -m llm_tutor.experiments.train_attention_seq2seq --epochs 2
uv run python -m llm_tutor.experiments.inspect_self_attention
uv run python -m llm_tutor.experiments.inspect_transformer_block
uv run python -m llm_tutor.experiments.train_mini_gpt --epochs 2 --batch-size 16 --block-size 32 --embed-dim 32 --num-heads 4 --num-layers 1 --sample-len 40
uv run python -m llm_tutor.experiments.inspect_gpt_loss
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
uv run python -m llm_tutor.experiments.train_mini_gpt --epochs 1 --batch-size 16 --block-size 32 --embed-dim 32 --num-heads 4 --num-layers 1 --sample-len 20 --checkpoint-path "$tmpdir/mini_gpt.pt"
uv run python -m llm_tutor.experiments.generate_with_mini_gpt --checkpoint-path "$tmpdir/mini_gpt.pt" --max-new-tokens 20 --top-k 5 --top-p 0.9
uv run python -m llm_tutor.experiments.train_sft --epochs 2 --batch-size 3 --embed-dim 32 --num-layers 1
uv run python -m llm_tutor.experiments.train_ppo_bandit --epochs 3
uv run python -m llm_tutor.experiments.train_dpo_bandit --epochs 3
uv run python -m llm_tutor.experiments.train_grpo_bandit --epochs 3 --group-size 4
uv run python -m llm_tutor.experiments.run_capstone_pipeline
rm -rf "$tmpdir"
