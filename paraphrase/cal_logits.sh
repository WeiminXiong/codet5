gpu_num=3
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 16))
  end_index=$(((i + 1) * 16))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$i python /home/weimin/CodeT5/CodeT5+/paraphrase/cal_logits_results.py --start_index ${start_index} --end_index ${end_index} --thread_id ${i}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
