import urllib.request, json, time

data = json.dumps({
    "model": "qwen",
    "messages": [{"role": "user", "content": "请写一首关于春天的短诗，4行即可"}],
    "max_tokens": 200,
    "stream": False
}).encode()

req = urllib.request.Request(
    "http://localhost:8080/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)

t0 = time.time()
resp = urllib.request.urlopen(req, timeout=120)
t1 = time.time()
result = json.loads(resp.read())
content = result["choices"][0]["message"]["content"]
usage = result.get("usage", {})

pt = usage.get("prompt_tokens", "?")
ct = usage.get("completion_tokens", "?")

print("=== 生成内容 ===")
print(content)
print()
print("=== 性能 ===")
print("总耗时: %.2fs" % (t1 - t0))
print("prompt tokens:", pt)
print("completion tokens:", ct)
if isinstance(ct, int) and ct > 0 and t1 > t0:
    print("生成速度: %.1f tok/s" % (ct / (t1 - t0)))
