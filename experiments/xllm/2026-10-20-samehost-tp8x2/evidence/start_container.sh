#!/usr/bin/env bash
set -euo pipefail

work=/data/tiankuan/huzhengyu/xllm-glm53-20261020
model=/data/tiankuan/wio/GLM-5.3-Flash-w8a8
runtime_root="$work/runtime-root"
image=quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911
container=xllm-glm53-hzy-171-20261020
vllm_container=vllm-glm53-hzy-171-20261020

test "$(docker inspect "$vllm_container" --format '{{index .Config.Labels "project"}}')" = hzy-vllm-glm53
docker stop -t 60 "$vllm_container"

if docker container inspect "$container" >/dev/null 2>&1; then
  test "$(docker inspect "$container" --format '{{index .Config.Labels "project"}}')" = hzy-xllm-glm53
  docker rm -f "$container"
fi

devices=()
for n in $(seq 0 15); do devices+=(--device "/dev/davinci$n:/dev/davinci$n"); done
for path in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc \
            /dev/infiniband/rdma_cm /dev/infiniband/umad0 /dev/infiniband/umad1 \
            /dev/infiniband/uverbs0 /dev/infiniband/uverbs1; do
  [[ -e "$path" ]] && devices+=(--device "$path:$path")
done

binds=(
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
  -v /etc/hccn.conf:/etc/hccn.conf:ro
  -v "$model:/models/glm53-flash:ro"
  -v "$work:/work:rw"
  -v "$runtime_root/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_xllm_math:/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_xllm_math:ro"
  -v "$runtime_root/usr/local/python3.11.15/lib/python3.11/site-packages/xlite:/usr/local/python3.11.15/lib/python3.11/site-packages/xlite:ro"
  -v "$runtime_root/usr/local/python3.11.15/lib/python3.11/site-packages/xlite-0.0.0.dist-info:/usr/local/python3.11.15/lib/python3.11/site-packages/xlite-0.0.0.dist-info:ro"
)
[[ -d /usr/local/Ascend/add-ons ]] && binds+=(-v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro)

docker run -d --name "$container" --label project=hzy-xllm-glm53 \
  --network host --shm-size 64g --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  "${devices[@]}" "${binds[@]}" \
  -e PIP_NO_INDEX=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e ATB_MATMUL_SHUFFLE_K_ENABLE=1 \
  -e ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 \
  -e ATB_OPSRUNNER_KERNEL_CACHE_GLOBAL_COUNT=16 \
  "$image" bash -lc 'sleep infinity'

docker exec "$container" bash -lc '
  set -e
  test -x /work/sources/xllm/build/xllm/core/server/xllm
  source /work/ascend_env.sh
  export LD_LIBRARY_PATH="/usr/local/python3.11.15/lib/python3.11/site-packages/xlite/lib:$LD_LIBRARY_PATH"
  ldd /work/sources/xllm/build/xllm/core/server/xllm | tee /work/logs/xllm-ldd.txt
  ! grep -q "not found" /work/logs/xllm-ldd.txt
'

docker exec -d "$container" bash -lc 'exec bash /work/launch_replica.sh 0 > /work/logs/replica0-launch.log 2>&1'
docker exec -d "$container" bash -lc 'exec bash /work/launch_replica.sh 1 > /work/logs/replica1-launch.log 2>&1'
docker ps --filter "name=^/${container}$" --format '{{.Names}} {{.Status}} {{.Image}}'
