## MODIFIED Requirements

### Requirement: 推送到腾讯云 TCR

**RENAMED**: 推送到阿里云 ACR

**FROM**: 推送到腾讯云 TCR

**TO**: 推送到阿里云 ACR

## REMOVED Requirements

### Requirement: 推送到腾讯云 TCR

**Reason**: 腾讯云 TCR 国内用户反馈推送速度过慢，替换为阿里云 ACR

**Migration**: 使用阿里云 ACR 地址 `crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com` 替代 TCR，Secrets 使用 ACR_USERNAME 和 ACR_PASSWORD