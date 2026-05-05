# Runnable Code 测试页面

## 普通 Run 按钮（无参数）

```python runnable
print("Hello World - 只有 Run 按钮")
```

## Run + GPU 按钮（gpu 参数）

```python runnable gpu
print("Hello GPU - 有 Run 和 GPU 两个按钮")
```

## 只有 GPU 按钮（gpuonly 参数）

```python runnable gpuonly
print("Hello GPU Only - 只有 GPU 按钮，没有 Run 按钮")
```

## GPU + timeout 参数组合

```python runnable gpu timeout=60
print("GPU + timeout 组合测试")
```

## gpuonly + timeout 参数组合

```python runnable gpuonly timeout=120
print("gpuonly + timeout 组合测试 - 只有 GPU 按钮")
```