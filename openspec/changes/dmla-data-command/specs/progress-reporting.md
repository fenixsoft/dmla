# Progress Reporting

进度反馈能力，支持长时间训练任务向前端实时报告进度。

## 功能需求

### Python API

容器内 Python 代码通过 `dmla_progress` 模块报告进度：

```python
from dmla_progress import ProgressReporter

# 创建进度报告器
progress = ProgressReporter(
    total_steps=100,           # 总步数（如 epoch 数）
    description="训练 AlexNet"
)

# 更新进度
progress.update(
    step=45,                   # 当前步数
    message="Epoch 45: Loss=0.32, Acc=72.35%"
)

# 完成时标记
progress.complete(message="训练完成，最佳准确率: 85.2%")

# 错误时标记
progress.error(message="训练中断：CUDA 内存不足")
```

### 进度文件格式

进度信息写入 `/workspace/progress.json`：

```json
{
  "description": "训练 AlexNet",
  "total_steps": 100,
  "current_step": 45,
  "percent": 45.0,
  "message": "Epoch 45: Loss=0.32, Acc=72.35%",
  "status": "running",
  "start_time": "2026-05-02T10:00:00Z",
  "elapsed_seconds": 1800,
  "estimated_remaining": 2200
}
```

状态值：
- `running`: 正在执行
- `complete`: 成功完成
- `error`: 执行出错

### 前端显示

前端接收 SSE 进度更新事件，显示进度条：

```
训练 AlexNet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
进度: 45% | Epoch 45/100 | Loss=0.32, Acc=72.35%
已用时: 30分钟 | 预计剩余: 37分钟
```

## 技术实现

### 容器内模块

`dmla_progress.py` 实现逻辑：

```python
import json
import time
from pathlib import Path

PROGRESS_FILE = Path('/workspace/progress.json')

class ProgressReporter:
    def __init__(self, total_steps, description=""):
        self.total_steps = total_steps
        self.description = description
        self.start_time = time.time()
        self._write_progress(0, "starting", description)
    
    def update(self, step, message=""):
        percent = (step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        remaining = elapsed * (self.total_steps - step) / step if step > 0 else 0
        
        self._write_progress(step, "running", message, percent, elapsed, remaining)
    
    def complete(self, message=""):
        elapsed = time.time() - self.start_time
        self._write_progress(self.total_steps, "complete", message, 100.0, elapsed)
    
    def error(self, message=""):
        elapsed = time.time() - self.start_time
        self._write_progress(self.current_step, "error", message, None, elapsed)
    
    def _write_progress(self, step, status, message, percent=None, elapsed=0, remaining=0):
        data = {
            "description": self.description,
            "total_steps": self.total_steps,
            "current_step": step,
            "percent": percent,
            "message": message,
            "status": status,
            "start_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.start_time)),
            "elapsed_seconds": int(elapsed),
            "estimated_remaining": int(remaining)
        }
        PROGRESS_FILE.write_text(json.dumps(data, ensure_ascii=False))
```

### 宿主机轮询

`sandbox.js` 实现轮询逻辑：

```javascript
function startProgressPolling(container, clientSocket) {
  const interval = setInterval(() => {
    // 从容器读取 progress.json
    container.exec({
      Cmd: ['cat', '/workspace/progress.json'],
      AttachStdout: true
    }).then(exec => {
      // 解析并发送到前端
      clientSocket.emit('progress', progressData)
    })
  }, 2000)  // 每 2 秒轮询
  
  return interval
}
```

## 使用约束

- 进度更新频率建议每 epoch 一次，避免频繁 IO
- progress.json 文件最大 1KB，避免大文件写入
- 仅在 timeout > 60 或 unlimited 时启用进度轮询
- 前端超时后停止接收进度更新