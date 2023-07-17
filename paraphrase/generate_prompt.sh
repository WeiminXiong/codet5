gpu_num=10
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 65))
  end_index=$(((i + 1) * 65))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    python /home/weimin/CodeT5/CodeT5+/paraphrase/apply_transform_and_filter.py --start_index ${start_index} --end_index ${end_index} --thread_id ${i}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

python /home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt/combine.py