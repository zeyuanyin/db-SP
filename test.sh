#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT" || exit 1

torchrun --nproc_per_node=4 ./test/test_hybrid_attn.py --sp_ulysses_degree 4 --ring_impl_type "basic" --attn_impl "paro"

# torchrun --nproc_per_node=4 selector.py