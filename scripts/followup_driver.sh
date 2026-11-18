#!/bin/bash
# Unattended pipeline on the pod: smoke check -> pilot (Qwen, Llama) -> judge pilot -> main (4 models + OLMo 3 cells) -> judge main.
# Pre-registered inputs only (notes/followup_preregistration.md). Status lines go to logs/followup/status.
cd /workspace/ea && . /workspace/venv/bin/activate && export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=WARNING
mkdir -p logs/followup results/followup; ST=logs/followup/status
say() { echo "$(date -u +%FT%TZ) $*" >> $ST; }
QW=Qwen/Qwen2.5-7B-Instruct; LL=meta-llama/Llama-3.1-8B-Instruct; MI=mistralai/Mistral-7B-Instruct-v0.3; GE=google/gemma-2-9b-it; OL=allenai/OLMo-2-1124-7B-Instruct
until grep -qE "tok/s|Traceback" logs/followup/smoke_qwen.log; do sleep 10; done
if [ "$(wc -l < results/followup/smoke/qwen.jsonl 2>/dev/null || echo 0)" -lt 18 ]; then say "SMOKE FAILED - driver stopped"; exit 1; fi
say "smoke ok"
gen() { # split k model [cells...]
  S=$(basename $3 | tr A-Z a-z); sp=$1; k=$2; m=$3; shift 3
  python scripts/followup_generate.py --model $m --data data/followup/rendered/$sp.jsonl --out results/followup/$sp/$S.jsonl --k $k "$@" > logs/followup/gen_${sp}_$S.log 2>&1 \
   && say "$sp $S generated: $(wc -l < results/followup/$sp/$S.jsonl) replies; $(tail -1 logs/followup/gen_${sp}_$S.log)" || say "$sp $S GEN FAILED"
}
judge() { # split
  for f in results/followup/$1/*.jsonl; do
    b=$(basename $f); [[ $b == judged_* ]] && continue
    python scripts/followup_judge.py --responses $f --data data/followup/rendered/$1.jsonl --out results/followup/$1/judged_$b > logs/followup/judge_$1_${b%.jsonl}.log 2>&1 \
     && say "$1 judged $b" || { say "$1 judge fp8 failed on $b; trying 14B bf16"; python scripts/followup_judge.py --responses $f --data data/followup/rendered/$1.jsonl --out results/followup/$1/judged14_$b --judge Qwen/Qwen2.5-14B-Instruct --quant "" > logs/followup/judge14_$1_${b%.jsonl}.log 2>&1 && say "$1 judged14 $b" || say "$1 JUDGE FAILED $b"; }
  done
}
gen pilot 4 $QW; gen pilot 4 $LL; judge pilot; say "PILOT DONE"
gen main 4 $QW; gen main 4 $LL; gen main 4 $MI; gen main 4 $GE; gen main 4 $OL --cells C0 C7 C8
say "MAIN GENERATION DONE"; judge main; say "MAIN JUDGED - ALL DONE (pod idle, stop it)"
