# Task 3 Report: CI 工作流修改 (publish-docker.yml)

## Status

Completed without issues.

## Commits

```
474ab8a - 修改 publish-docker.yml CI 工作流，新增 FC 函数计算流水线
```

## Changes Summary

File modified: `.github/workflows/publish-docker.yml` (123 insertions, 1 deletion)

### New jobs added (in order between `push-acr` and `summary`):

1. **build-fc** (lines 178-225): Builds FC slim image from `Dockerfile.sandbox.fc`, runs health check test, uploads artifact. Depends on `build-images`.

2. **push-acr-fc** (lines 227-270): Downloads FC artifact, logs into ACR, tags and pushes `:fc` image. Gated by `skip_acr` input and `build-fc` success.

3. **update-fc** (lines 272-294): Installs aliyun CLI, configures credentials from secrets, calls `aliyun fc update-function` to update the `sandbox-cpu` function image. Depends on both `push-acr` and `push-acr-fc`.

### Modified job:

- **summary** (line 298): `needs` updated to `[build-images, build-fc, push-dockerhub, push-acr, push-acr-fc, update-fc]`.
- **summary** (lines 316-318): Added FC image output to `$GITHUB_STEP_SUMMARY`.

## Test Summary

No automated tests for CI workflow changes. Verified by reading the final file:
- YAML structure is valid
- Job dependency chain is correct
- Conditional expressions match existing patterns (`always()` + input check + upstream result check)

## Concerns

1. The `Dockerfile.sandbox.fc` must exist for `build-fc` to succeed. The brief assumes it exists.
2. The `ALIBABA_CLOUD_ACCESS_KEY_ID` and `ALIBABA_CLOUD_ACCESS_KEY_SECRET` secrets must be configured in GitHub repository settings.
3. The `update-fc` job's `needs` includes `push-acr` even though it doesn't directly use its output — this ensures the regular ACR push completes before FC update, which is correct for the design intent (FC update should happen after all pushes).
