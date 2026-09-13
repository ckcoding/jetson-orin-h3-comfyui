#!/usr/bin/env python3
"""10分钟并发压测: llama-server (Qwen2.5-3B) @ 192.168.8.111:8080"""
import urllib.request, json, time, threading, glob, random, os

DURATION = 600       # 10 分钟
CONCURRENCY = 2
URL = "http://localhost:8080/v1/chat/completions"

PROMPTS = [
    "请用100字介绍一下人工智能的发展历史",
    "写一首关于大海的五言绝句",
    "解释一下什么是量子计算，200字以内",
    "把这段话翻译成英文：今天天气真好，我们去公园散步吧。",
    "帮我写一封请假短信给老板，理由是家里有事",
    "1到100的质数有多少个？列举前10个",
    "简单介绍下中国的四大发明",
    "用python写一个快速排序，带注释",
    "讲一个简短的笑话",
    "分析一下电动车和燃油车的优缺点，各50字",
    "把这句话改写成商务语气：这个东西太贵了，便宜点呗",
    "请列出5个健康的生活习惯",
]

stop_time = time.time() + DURATION
lock = threading.Lock()
stats = {"req": 0, "tokens": 0, "err": 0, "lat": []}

def get_temps():
    temps = []
    for z in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
        try:
            temps.append(int(open(z).read().strip()) / 1000)
        except Exception:
            pass
    return max(temps) if temps else 0

def mem_used_mb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
    return (info["MemTotal"] - info["MemAvailable"]) // 1024

def worker(wid):
    while time.time() < stop_time:
        prompt = random.choice(PROMPTS)
        data = json.dumps({
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150, "temperature": 0.7, "stream": False
        }).encode()
        req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            r = json.loads(resp.read())
            dt = time.time() - t0
            n = r.get("usage", {}).get("completion_tokens", 0)
            with lock:
                stats["req"] += 1
                stats["tokens"] += n
                stats["lat"].append(dt)
        except Exception as e:
            with lock:
                stats["err"] += 1
            time.sleep(1)

def monitor():
    while time.time() < stop_time:
        time.sleep(30)
        with lock:
            n, tok, err = stats["req"], stats["tokens"], stats["err"]
        el = DURATION - (stop_time - time.time())
        print("[%3ds] 请求=%d 错误=%d 吞吐=%.1f tok/s 温度=%.1fC 内存=%dMB" %
              (el, n, err, tok / max(el, 1), get_temps(), mem_used_mb()), flush=True)

threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(CONCURRENCY)]
m = threading.Thread(target=monitor, daemon=True)
m.start()
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()

el = time.time() - t0
lat = sorted(stats["lat"])
def pct(p):
    return lat[int(len(lat) * p)] if lat else 0

print("\n" + "=" * 50)
print("压测结束 总耗时 %.0fs" % el)
print("总请求数: %d  失败: %d  失败率: %.2f%%" % (stats["req"], stats["err"], 100.0 * stats["err"] / max(stats["req"] + stats["err"], 1)))
print("总生成 tokens: %d" % stats["tokens"])
print("平均吞吐: %.1f tok/s | 每请求平均 %.1f tok/s" % (stats["tokens"] / el, (stats["tokens"] / max(stats["req"], 1)) / max(el / max(stats["req"], 1), 0.001)))
print("平均每请求耗时: %.2fs" % (sum(lat) / max(len(lat), 1)))
print("P50: %.2fs  P95: %.2fs  P99: %.2fs  最大: %.2fs" % (pct(0.5), pct(0.95), pct(0.99), max(lat) if lat else 0))
print("最终温度: %.1fC  内存: %dMB" % (get_temps(), mem_used_mb()))
print("=" * 50)
