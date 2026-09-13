import urllib.request, json, time

# 测试首字延迟（stream模式）
data = json.dumps({
    "model": "qwen",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50,
    "stream": True
}).encode()

req = urllib.request.Request(
    "http://localhost:8080/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)

t0 = time.time()
resp = urllib.request.urlopen(req, timeout=60)
first_token_time = None
token_count = 0
last_token_time = t0

for line in resp:
    line = line.decode().strip()
    if line.startswith("data: ") and line != "data: [DONE]":
        if first_token_time is None:
            first_token_time = time.time()
            print("首字延迟: %.3fs" % (first_token_time - t0))
        token_count += 1
        last_token_time = time.time()

t_end = time.time()
print("总耗时: %.3fs" % (t_end - t0))
print("token数: %d" % token_count)
if token_count > 0:
    gen_time = t_end - first_token_time
    print("生成速度: %.1f tok/s" % (token_count / gen_time))
