#!/usr/bin/env bash
#!/usr/bin/env bash
# 运行预设的若干实验组合
# 用法：
#   chmod +x run_lambda_uncert.sh
#   ./run_lambda_uncert.sh

set -euo pipefail

log_dir="效果"
mkdir -p "${log_dir}"

run_cmd() {
  local name="$1"; shift
  local log_path="${log_dir}/${name}.log"
  echo "==> Running ${name} => log: ${log_path}"
  echo "CMD: $*" > "${log_path}"
  "$@" >> "${log_path}" 2>&1
}

# 1) tiny_imagenet, buffer 2000, lambda 5 seed 4
run_cmd "tiny_imagenet_buf2000_lam5" \
  python main.py \
    --dataset tiny_imagenet \
    --buffer_size 2000 \
    --method mose \
    --seed 4 \
    --run_nums 1\
    --gpu_id 0 \
    --n_tasks 100 \
    --lambda_uncert 5

# 2) tiny_imagenet, buffer 1000, lambda 5
run_cmd "tiny_imagenet_buf1000_lam5" \
  python main.py \
    --dataset tiny_imagenet \
    --buffer_size 1000\
    --method mose \
    --seed 0 \
    --run_nums 5 \
    --gpu_id 0 \
    --lambda_uncert 5
# 2) tiny_imagenet, buffer 500mbda 5
run_cmd "tiny_imagenet_buf500_lam5" \
  python main.py \
    --dataset tiny_imagenet \
    --buffer_size 500\
    --method mose \
    --seed 0 \
    --run_nums  5 \
    --gpu_id 0 \
    --lambda_uncert 5


echo "全部任务完成，日志保存在 ${log_dir}"

