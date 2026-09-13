#!/bin/bash
# 重组被分卷的大文件: 在仓库根目录执行
while read f; do
  mkdir -p "$(dirname "$f")"
  cat bigfiles/$f.part_* > "$f"
  echo "rebuilt: $f"
done < bigfiles.list
