# Source inside the pinned xLLM container, including non-interactive docker exec.
_xllm_had_nounset=0
case $- in *u*) _xllm_had_nounset=1; set +u ;; esac
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh ]; then
 source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
fi
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
# These exports are in the image's .bashrc, not in its Docker ENV metadata.
# Preserve the same vendor priority for non-interactive commands.
for _xllm_vendor in custom_xllm_math custom_transformer glm_next_transformer kpool_transformer; do
 _xllm_vendor_path=/usr/local/Ascend/cann-9.0.0/opp/vendors/$_xllm_vendor
 if [ ! -d "$_xllm_vendor_path" ]; then
  if [ "$_xllm_vendor" = custom_xllm_math ]; then continue; fi
  echo "Missing required OPP vendor: $_xllm_vendor_path" >&2
  return 1
 fi
 export ASCEND_CUSTOM_OPP_PATH="$_xllm_vendor_path:${ASCEND_CUSTOM_OPP_PATH:-}"
done
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_xllm_math/op_api/lib:${LD_LIBRARY_PATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export XLLM_ASCEND_ENV_READY=1
if [ "$_xllm_had_nounset" = 1 ]; then set -u; fi
unset _xllm_had_nounset _xllm_vendor _xllm_vendor_path
