#!/bin/bash
API=http://192.168.8.111:8080
OUT=/d/ComfyUI_windows_portable/workspace/cap_results.txt
CAPDIR=/d/ComfyUI_windows_portable/workspace/capjson
mkdir -p "$CAPDIR"

declare -a NAMES=("MATH" "LOGIC" "KNOWLEDGE" "CODE" "CREATIVE")
declare -a KEYS=("math" "logic" "knowledge" "code" "creative")
declare -a PROMPTS=(
"Xiao Ming has 37 apples, gives 9 to Xiao Hong, then buys 5 boxes of 12 apples each, and splits all apples equally among 4 friends. How many does each friend get? Show the calculation."
"All cats fear water. Tom is a cat. Does Tom fear water? Answer in two sentences with reasoning."
"Explain what Mixture of Experts (MoE) architecture is and why it speeds up inference. Under 120 words."
"Write a Python function that checks if a string is a palindrome, ignoring case and punctuation. Code only."
"Write a 4-line poem using the words: moon, coffee, code."
)

for MODEL in qwen38 qwen36; do
  if [ "$MODEL" = "qwen38" ]; then TAG="27B-dense"; else TAG="35B-A3B-MoE"; fi
  echo "########## $TAG ##########" >> "$OUT"
  for idx in 0 1 2 3 4; do
    NAME=${NAMES[$idx]}; KEY=${KEYS[$idx]}; PR=${PROMPTS[$idx]}
    T0=$(date +%s)
    curl -s --max-time 400 $API/api/generate -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"$PR\",\"stream\":false,\"options\":{\"num_predict\":300,\"temperature\":0.3}}" \
      -o "$CAPDIR/cap_${TAG}_${KEY}.json"
    T1=$(date +%s)
    F="$CAPDIR/cap_${TAG}_${KEY}.json"
    python - "$F" "$TAG" "$NAME" "$T0" "$T1" >> "$OUT" 2>&1 << 'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding='utf-8'))
    ec = d.get('eval_count', 0); ed = d.get('eval_duration', 1)
    r = d.get('response', '').replace('\n', ' | ')[:200]
    speed = ec / (ed / 1e9) if ed else 0
    print(f"[{sys.argv[2]}][{sys.argv[3]}] {ec}tok {int(sys.argv[5])-int(sys.argv[4])}s | {speed:.1f} tok/s")
    print("   >> " + r)
except Exception as e:
    print(f"[{sys.argv[2]}][{sys.argv[3]}] 解析失败: {e}")
PYEOF
  done
done
echo "CAP TEST DONE" >> "$OUT"
