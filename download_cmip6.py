# -*- coding: utf-8 -*-
"""
CMIP6 未来气候数据批量下载器 (Science Data Bank / scidb.cn)
在【你自己的电脑】上运行(不是在助手沙箱里——沙箱无法访问外网)。

用途:下载吉林水稻未来预测所需的 tas(月均温) + pr(月降水),
      SSP2-4.5 / SSP5-8.5,历史基准1991-2020 + 未来2041-2070。

依赖:  pip install requests
运行:  python download_cmip6.py
断点续传:中途断了直接再跑一次即可,已下好的自动跳过、下了一半的自动续传。
"""
import os, re, sys, time
import requests

# ================== 配置(按需修改) ==================
# 用哪个清单:先跑 ensemble(6个文件,快),要不确定性再换 full(42个文件)
URL_LIST = r"D:\水稻\增温驱动吉林水田扩张与增产_1985-2020\02_中间数据\cmip6\cmip6_urls_ensemble.txt"
# URL_LIST = r"D:\水稻\增温驱动吉林水田扩张与增产_1985-2020\02_中间数据\cmip6\cmip6_urls_full.txt"

OUT_DIR  = r"D:\水稻\增温驱动吉林水田扩张与增产_1985-2020\02_中间数据\cmip6"
MAX_RETRY = 5          # 每个文件最多重试次数
TIMEOUT   = 60         # 连接/读取超时(秒)
CHUNK     = 1024 * 512 # 512KB 分块
# ====================================================

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CMIP6-downloader"}

def sub_dir(fname):
    """根据文件名归类到子目录。"""
    if "1991-2020" in fname:
        return os.path.join(OUT_DIR, "baseline_1991-2020")
    scen = "SSP245" if "SSP245" in fname else ("SSP585" if "SSP585" in fname else "other")
    period = "2041-2070" if "2041-2070" in fname else "future"
    return os.path.join(OUT_DIR, period, scen)

def parse_lines(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("http"):
                continue
            m = re.search(r"fileName=([^&]+)", line)
            fname = m.group(1) if m else line.split("/")[-1].split("?")[0]
            items.append((line, fname))
    return items

def download_one(url, fname):
    dst_dir = sub_dir(fname); os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, fname)
    tmp = dst + ".part"
    # 已完成?比对远端大小
    try:
        h = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        total = int(h.headers.get("Content-Length", 0))
    except Exception:
        total = 0
    if os.path.exists(dst) and total and os.path.getsize(dst) == total:
        print(f"  [跳过] 已完整: {fname}"); return True

    for attempt in range(1, MAX_RETRY + 1):
        pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = dict(HEADERS)
        if pos and total:
            headers["Range"] = f"bytes={pos}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                mode = "ab" if (pos and r.status_code == 206) else "wb"
                if mode == "wb":
                    pos = 0
                grand = total or int(r.headers.get("Content-Length", 0)) + pos
                t0 = time.time(); done = pos
                with open(tmp, mode) as fo:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fo.write(chunk); done += len(chunk)
                        if grand:
                            pct = 100 * done / grand
                            sp = done / 1e6 / max(time.time() - t0, 1e-3)
                            print(f"\r  {fname[:48]:48s} {pct:5.1f}%  {done/1e6:7.1f}MB  {sp:5.1f}MB/s", end="")
                print()
            os.replace(tmp, dst)
            if total and os.path.getsize(dst) != total:
                raise RuntimeError("大小不符,重下")
            print(f"  [完成] {fname}")
            return True
        except Exception as e:
            print(f"\n  [重试 {attempt}/{MAX_RETRY}] {fname}: {e}")
            time.sleep(min(5 * attempt, 30))
    print(f"  [失败] {fname} —— 请稍后重跑脚本(会自动续传)")
    return False

def main():
    if not os.path.exists(URL_LIST):
        print("找不到清单文件:", URL_LIST); sys.exit(1)
    items = parse_lines(URL_LIST)
    print(f"共 {len(items)} 个文件,输出到 {OUT_DIR}\n")
    ok = 0
    for i, (url, fname) in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {fname}")
        if download_one(url, fname):
            ok += 1
    print(f"\n完成 {ok}/{len(items)}。如有失败,直接再次运行本脚本即可续传。")

if __name__ == "__main__":
    main()
