import { path } from 'vuepress/utils'
import { defaultTheme } from '@vuepress/theme-default'

export const dmlaTheme = (options = {}) => {
  return {
    name: 'vuepress-theme-dmla',
    extends: defaultTheme(options),
    clientConfigFile: path.resolve(__dirname, './client.js'),
    // 覆盖默认的 useRelatedLinks，实现跨组导航
    alias: {
      '@theme/useRelatedLinks': path.resolve(__dirname, './composables/useRelatedLinks.js'),
    },
  }
}

export default dmlaTheme