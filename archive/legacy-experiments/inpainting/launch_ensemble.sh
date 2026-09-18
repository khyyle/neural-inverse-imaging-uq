#!/usr/bin/env bash
NG=4
N=100
echo "starting job"
for SEED in $(seq 0 $((N-1))); do
  GPU=$(( SEED % NG ))
  echo "▶︎ seed=${SEED} → GPU=${GPU}"
  CUDA_VISIBLE_DEVICES=${GPU} \
    python run_inpaint.py ${SEED} \
    > logs/seed_${SEED}.log 2>&1 &

  while [ $(jobs -r | wc -l) -ge ${NG} ]; do sleep 1; done
done

wait
echo "All ${N} runs complete."
