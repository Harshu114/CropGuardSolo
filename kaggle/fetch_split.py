"""Download only cropguard_v7_split/ out of the pack kernel's output, in parallel.

`kaggle kernels output` enumerates every output file (6558 negatives + the split
~2500 files) 20 per API page and downloads each one synchronously; the negatives
make it crawl and it stalls. This enumerates with big pages, filters to the
split, then pulls the files with a thread pool -- the per-file latency is the
bottleneck, not bandwidth.

  python3 kaggle/fetch_split.py ~/tmp/cgpack [workers]
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

KERNEL = "ganrajlongadge/cropguard-v7-plantseg-pack"
PATTERN = "cropguard_v7_split/"


def list_matching(api):
    owner, slug, _ = api.parse_kernel_string(KERNEL)
    token, items = None, []
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        req.page_size = 100          # 200 hangs the endpoint; 100 is ~0.9 s/page
        if token:
            req.page_token = token
        for attempt in range(3):
            try:
                with api.build_kaggle_client() as kaggle:
                    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                break
            except Exception as e:                      # noqa: BLE001
                print(f"  page retry {attempt}: {e}", flush=True)
        else:
            raise RuntimeError("enumeration failed")
        items += [f for f in (resp.files or []) if PATTERN in f.file_name]
        token = getattr(resp, "next_page_token", None)
        print(f"  enumerated {len(items)} matching so far", flush=True)
        if not token:
            break
    return items


def fetch(item, dest, session):
    out = os.path.join(dest, item.file_name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = session.get(item.url, timeout=120)
    r.raise_for_status()
    with open(out, "wb") as f:
        f.write(r.content)
    return len(r.content)


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tmp/cgpack")
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    api = KaggleApi()
    api.authenticate()
    items = list_matching(api)
    print(f"{len(items)} files to fetch, {workers} workers", flush=True)

    sess = requests.Session()
    done = total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, it, dest, sess) for it in items]
        for fu in as_completed(futs):
            try:
                total += fu.result()
            except Exception as e:                      # noqa: BLE001
                print("  FAIL:", e, flush=True)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(items)} files, {total/1e6:.1f} MB", flush=True)
    print(f"DONE: {done}/{len(items)} files, {total/1e6:.1f} MB -> {dest}")


if __name__ == "__main__":
    main()
