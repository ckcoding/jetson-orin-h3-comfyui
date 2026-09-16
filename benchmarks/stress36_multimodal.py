#!/usr/bin/env python3
# qwen36 多模态压测: 图片+文本, 1小时, 记录速度/内容/错误
import json, base64, urllib.request, urllib.error, time, sys, os

API = "http://192.168.8.111:8080"
DURATION = 3600  # 1小时
IMGS = [r"D:/ComfyUI_windows_portable/workspace/visiontest/crane.png",
        r"D:/ComfyUI_windows_portable/workspace/visiontest/car.png",
        r"D:/ComfyUI_windows_portable/workspace/visiontest/ramen.png"]
QUESTIONS = [
    "详细描述这张图片的内容",
    "这张图的主色调是什么？",
    "图片里有哪些物体？列出三个",
    "这张照片的光线条件如何？",
    "如果在画面右侧加一个杯子，构图会怎样？",
    "这张图给人的情绪感受是什么？",
]
LOG = r"D:/ComfyUI_windows_portable/workspace/stress36_log.jsonl"
SUMMARY = r"D:/ComfyUI_windows_portable/workspace/stress36_summary.txt"

b64s = [base64.b64encode(open(p, 'rb').read()).decode() for p in IMGS]
stats = {"ok": 0, "err": 0, "tokens": 0, "gen_time": 0.0, "start": time.time()}
idx = 0

def write_summary(el, note=""):
    with open(SUMMARY, 'w', encoding='utf-8') as s:
        s.write(f"进度 {el/60:.1f}/60 min | 请求 {idx} | 成功 {stats['ok']} | 失败 {stats['err']} | "
                f"累计 {stats['tokens']} tok | 平均 {stats['tokens']/max(stats['gen_time'],0.01):.1f} tok/s | {note}\n")

with open(LOG, 'a', encoding='utf-8') as log:
    while time.time() - stats["start"] < DURATION:
        idx += 1
        img_i = idx % 3
        q = QUESTIONS[idx % len(QUESTIONS)]
        payload = {"model": "qwen3.6-35b-a3b", "stream": False,
                   "messages": [{"role": "user", "content": q, "images": [b64s[img_i]]}],
                   "options": {"num_predict": 250, "temperature": 0.7}}
        t0 = time.time()
        entry = {"req": idx, "img": os.path.basename(IMGS[img_i]), "q": q[:30],
                 "t": time.strftime('%H:%M:%S'), "elapsed_min": round((t0-stats["start"])/60, 1)}
        try:
            body = json.dumps(payload).encode()
            r = urllib.request.urlopen(urllib.request.Request(
                API + "/api/chat", data=body,
                headers={"Content-Type": "application/json"}), timeout=600)
            d = json.loads(r.read().decode('utf-8'))
            m = d.get("message", {})
            ec, ed = d.get("eval_count", 0), d.get("eval_duration", 0)
            speed = ec / (ed / 1e9) if ed else 0
            content = (m.get("content") or "")
            reasoning = (m.get("reasoning") or m.get("thinking") or "")
            entry.update({"ok": True, "tok": ec,
                          "ms_per_tok": round(ed/1e6/max(ec,1),1),
                          "tok_s": round(speed,1),
                          "wall": round(time.time()-t0,1),
                          "content": content[:120],
                          "think_len": len(reasoning)})
            stats["ok"] += 1; stats["tokens"] += ec; stats["gen_time"] += ed/1e9
        except Exception as e:
            entry.update({"ok": False, "error": str(e)[:200]})
            stats["err"] += 1
        log.write(json.dumps(entry, ensure_ascii=False) + "\n"); log.flush()
        el = time.time() - stats["start"]
        write_summary(el)
        # 错误后暂停30s让服务恢复
        if not entry.get("ok"):
            time.sleep(30)

el = time.time() - stats["start"]
write_summary(el, "=== 压测结束 ===")
with open(SUMMARY, 'a', encoding='utf-8') as s:
    s.write(f"总请求 {idx} | 成功率 {stats['ok']/max(idx,1)*100:.1f}% | "
            f"总 token {stats['tokens']} | 平均生成速度 {stats['tokens']/max(stats['gen_time'],0.01):.1f} tok/s | "
            f"总时长 {(time.time()-stats['start'])/60:.1f} min\n")
