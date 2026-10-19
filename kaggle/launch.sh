#!/usr/bin/env bash
# CropGuard v7 kernel launcher.
#
#   push-code    rebuild /tmp/v7code FROM THE REPO and push a new code dataset
#                version. Always do this before pushing any kernel.
#   smoke        ~6 min, validates mount + v7 code path (run this first!)
#   train        full v7 on the SAME corpus as v6 (comparable numbers)
#   derive       PlantSeg -> corpus ONLY. NO GPU => burns ZERO GPU quota.
#                Publish its output as `cropguard-v7-plantseg-corpus`.
#   plantseg     full v7 on the PlantSeg single-dataset corpus. Attaches the
#                published corpus, so it skips the 3 GB download/derive.
#   status       status of every cropguard kernel on the account
#   out <name>   pull a kernel's output to /tmp/kout
#
# GPU note: a Kaggle account may run at most 2 GPU sessions at once. `derive`
# and `smoke` may overlap safely (derive takes no GPU slot).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  push-code)
    "$REPO/kaggle/build_code_dataset.sh" /tmp/v7code
    kaggle datasets version -p /tmp/v7code -m "${2:-v7 code refresh}" --dir-mode zip ;;
  smoke)    kaggle kernels push -p "$REPO/kaggle/smoke" ;;
  train)    kaggle kernels push -p "$REPO/kaggle" ;;
  derive)   kaggle kernels push -p "$REPO/kaggle/plantseg-derive" ;;
  pack)     kaggle kernels push -p "$REPO/kaggle/plantseg-pack" ;;
  plantseg) kaggle kernels push -p "$REPO/kaggle/plantseg" ;;
  status)
    for k in smoke:ganrajlongadge/cropguard-v7-smoke \
             train:ganrajlongadge/cropguard-v7-train \
             derive:ganrajlongadge/cropguard-v7-plantseg-derive \
             pack:ganrajlongadge/cropguard-v7-plantseg-pack \
             plantseg:ganrajlongadge/cropguard-v7-plantseg; do
      printf "%-9s " "${k%%:*}"
      kaggle kernels status "${k##*:}" 2>&1 | tail -1
    done ;;
  out)
    kaggle kernels output "ganrajlongadge/cropguard-v7-${2:-smoke}" -p /tmp/kout ;;
  *)
    echo "usage: $0 {push-code [msg]|smoke|train|derive|pack|plantseg|status|out <name>}"
    exit 1 ;;
esac
