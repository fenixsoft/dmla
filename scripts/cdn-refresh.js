/**
 * 腾讯云 CDN 刷新脚本
 * 在部署后自动刷新 CDN 缓存
 */

async function refreshCDN() {
  // 验证环境变量
  const secretId = process.env.TENCENT_SECRET_ID
  const secretKey = process.env.TENCENT_SECRET_KEY
  const cdnDomain = process.env.CDN_DOMAIN

  if (!secretId || !secretKey || !cdnDomain) {
    console.log('⚠️  腾讯云 CDN 配置不完整，跳过刷新')
    console.log('需要配置以下 GitHub Secrets:')
    console.log('  - TENCENT_SECRET_ID')
    console.log('  - TENCENT_SECRET_KEY')
    console.log('  - CDN_DOMAIN')
    return
  }

  try {
    // 动态导入腾讯云 CDN SDK
    const tencentcloud = await import('tencentcloud-sdk-nodejs-cdn')
    const CdnClient = tencentcloud.cdn.v20180606.Client

    // 创建客户端
    const client = new CdnClient({
      credential: {
        secretId,
        secretKey
      },
      region: '',
      profile: {
        httpProfile: {
          endpoint: 'cdn.tencentcloudapi.com'
        }
      }
    })

    // 全站刷新 - 刷新根目录和 assets 目录
    console.log(`🚀 正在刷新 CDN: https://${cdnDomain}/`)

    const paths = [
      `https://${cdnDomain}/`,           // 根目录
      `https://${cdnDomain}/assets/`     // assets 目录（JS/CSS 文件）
    ]

    const result = await client.PurgePathCache({
      Paths: paths,
      FlushType: 'flush'
    })

    console.log('✅ CDN 刷新任务已提交:')
    console.log(`   任务 ID: ${result.TaskId}`)
    console.log(`   刷新路径:`)
    paths.forEach(p => console.log(`     - ${p}`))

  } catch (error) {
    console.error('❌ CDN 刷新失败:', error.message)
    // 不抛出错误，避免中断部署流程
  }
}

// 执行刷新
refreshCDN()