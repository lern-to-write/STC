#!/bin/bash
# 监控脚本：查看你的程序优先级和资源使用

echo "=== 进程信息 ==="
pgrep -af "eval.py" | while read line; do
    PID=$(echo $line | awk '{print $1}')
    echo "进程PID: $PID"
    echo "  OOM Score: $(cat /proc/$PID/oom_score 2>/dev/null || echo 'N/A')"
    echo "  OOM Adj: $(cat /proc/$PID/oom_score_adj 2>/dev/null || echo 'N/A')"
    echo "  Nice值: $(ps -o ni= -p $PID 2>/dev/null || echo 'N/A')"
    echo "  内存使用: $(ps -o %mem= -p $PID 2>/dev/null || echo 'N/A')%"
    echo ""
done

echo "=== 系统内存状态 ==="
free -h
echo ""
echo "=== 前10内存使用进程 ==="
ps aux --sort=-%mem | head -11