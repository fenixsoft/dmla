/**
 * VuePress 评论插件
 * 基于 GitHub Issues 实现
 */
import { path } from 'vuepress/utils'
import { getDirname } from 'vuepress/utils'

const __dirname = getDirname(import.meta.url)

export const commentsPlugin = (options = {}) => {
  const {
    repo,           // 仓库: 'owner/repo'
    clientId,       // GitHub OAuth Client ID
    proxyUrl        // OAuth 代理 URL (可选)
  } = options

  return {
    name: 'vuepress-plugin-comments',

    define: {
      __COMMENTS_REPO__: repo || '',
      __COMMENTS_CLIENT_ID__: clientId || '',
      __COMMENTS_PROXY_URL__: proxyUrl || ''
    },

    clientConfigFile: path.resolve(__dirname, 'client.js')
  }
}

export default commentsPlugin