/**
 * 腾讯云 CDN 刷新与预热脚本
 * 在部署后自动刷新 CDN 缓存，并预热关键页面
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

    // ========== 第一步：刷新缓存 ==========
    console.log(`🚀 正在刷新 CDN: https://${cdnDomain}/`)

    const paths = [
      `https://${cdnDomain}/`,           // 根目录
      `https://${cdnDomain}/assets/`     // assets 目录（JS/CSS 文件）
    ]

    const flushResult = await client.PurgePathCache({
      Paths: paths,
      FlushType: 'flush'
    })

    console.log('✅ CDN 刷新任务已提交:')
    console.log(`   任务 ID: ${flushResult.TaskId}`)
    console.log(`   刷新路径:`)
    paths.forEach(p => console.log(`     - ${p}`))

    // ========== 第二步：预热关键页面 ==========
    console.log('\n🔥 正在预热关键页面...')

    // 预热的关键 URL 列表（首页及常用页面）
    const urlsToPush = [
      // 首页及主要导航页面
      `https://${cdnDomain}/`,
      `https://${cdnDomain}/index.html`,
      // 深度学习核心路径（高频访问）
      `https://${cdnDomain}/deep-learning/`,
      `https://${cdnDomain}/deep-learning/neural-network-optimization/`,
      `https://${cdnDomain}/deep-learning/neural-network-optimization/activation-functions.html`,
      `https://${cdnDomain}/deep-learning/neural-network-optimization/loss-functions.html`,
      `https://${cdnDomain}/deep-learning/neural-network-optimization/adaptive-optimizers.html`,
      // 机器学习基础路径
      `https://${cdnDomain}/statistical-learning/`,
      `https://${cdnDomain}/statistical-learning/linear-models/`,
      // 其他高频页面
      `https://${cdnDomain}/404.html`
    ]

    // 分批预热（每次最多 500 条）
    const batchSize = 500
    for (let i = 0; i < urlsToPush.length; i += batchSize) {
      const batch = urlsToPush.slice(i, i + batchSize)

      const pushResult = await client.PushUrlsCache({
        Urls: batch,
        Area: 'mainland'  // 预热至境内节点
      })

      console.log('✅ CDN 预热任务已提交:')
      console.log(`   任务 ID: ${pushResult.TaskId}`)
      console.log(`   预热 URL (${i + 1}-${i + batch.length}/${urlsToPush.length}):`)
      batch.forEach(u => console.log(`     - ${u}`))
    }

    console.log('\n🎉 CDN 刷新与预热完成！')
    console.log('💡 提示: 预热任务需要一定时间生效，请稍后验证访问速度')

  } catch (error) {
    console.error('❌ CDN 操作失败:', error.message)
    if (error.code) {
      console.error(`   错误码: ${error.code}`)
    }
    // 不抛出错误，避免中断部署流程
  }
}

// 执行刷新与预热
refreshCDN()