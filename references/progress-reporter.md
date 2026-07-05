# 进度报告机制（ProgressReporter）

长时间训练任务可通过 `ProgressReporter` 向前端报告实时进度，前端通过轮询 `/workspace/progress.json` 获取进度信息并显示进度条。

**模块位置**：`local-server/src/dmla_progress.py`

**使用方式**：

```python
from dmla_progress import ProgressReporter

# 初始化进度报告器
progress = ProgressReporter(
    total_steps=100,        # 总步数（如 epoch 数）
    description="训练 AlexNet"
)

# 训练循环中更新进度
for epoch in range(100):
    train_one_epoch()
    loss = calculate_loss()
    progress.update(
        step=epoch + 1,
        message=f"Epoch {epoch+1}: Loss={loss:.4f}",
        extra_data={"loss": loss, "accuracy": accuracy}
    )

# 训练完成
progress.complete(
    message="训练完成",
    extra_data={"final_accuracy": 0.85}
)
```

**进度数据结构**：

```json
{
  "description": "训练 AlexNet",
  "total_steps": 100,
  "current_step": 50,
  "percent": 50.0,
  "message": "Epoch 50: Loss=0.32",
  "status": "running",
  "start_time": "2026-05-02T12:00:00Z",
  "elapsed_seconds": 120,
  "estimated_remaining": 120,
  "extra_data": {"loss": 0.32, "accuracy": 0.78}
}
```

**状态类型**：
- `starting`: 任务开始
- `running`: 任务进行中
- `complete`: 任务完成
- `error`: 任务出错

**辅助函数**：

```python
from dmla_progress import get_progress, clear_progress

# 读取当前进度
progress = get_progress()

# 清除进度文件
clear_progress()
```

**注意事项**：
- 进度文件路径固定为 `/workspace/progress.json`
- 写入失败不会影响训练，仅打印警告
- `extra_data` 可携带自定义指标（loss、accuracy 等）
