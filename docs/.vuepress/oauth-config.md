# GitHub OAuth 配置

## OAuth App 信息

- **Application name**: DMLA Comments
- **Homepage URL**: https://ai.icyfenix.cn
- **Authorization callback URL**:
  - https://fenixsoft.github.io/dmla/
  - https://ai.icyfenix.cn
  - http://localhost:8080
  - http://localhost:8081
  - http://localhost:8081

## 凭证

- **Client ID**: Ov23liQbtfoZGMcsU9VV
- **Client Secret**: f93311c59cbe51af671a3aa95af64028d4d7d1d3

## 代理服务器部署

OAuth 代理用于安全地交换 access_token（clientSecret 不能暴露在前端）。

### 腾讯云云函数部署步骤

1. 登录 [腾讯云控制台](https://console.cloud.tencent.com/)
2. 进入 **云函数 SCF** → **函数服务**
3. 点击 **新建**，选择 **自定义创建**
4. 运行环境选择 **Node.js 18** 或更高版本
5. 代码上传方式选择 **本地上传**，上传 `scripts/oauth-proxy-worker.js`
6. 触发器配置选择 **API 网关触发器**，启用集成响应
7. 点击 **完成** 部署
8. 部署后在触发器管理中获取访问 URL

### 代理 URL 格式

部署成功后，代理 URL 类似：
```
https://service-xxx.gz.apigw.tencentcs.com/release/callback
```

### 测试代理

```bash
# 健康检查
curl https://your-proxy-url/

# 测试 token 交换（需要有效的 code）
curl "https://your-proxy-url/callback?code=YOUR_CODE"
```