import { defaultTheme } from '@vuepress/theme-default'
import { getDirname, path } from 'vuepress/utils'

const __dirname = getDirname(import.meta.url)

export default {
  extends: defaultTheme,

  layouts: {
    Layout: path.resolve(__dirname, 'layouts/Layout.vue')
  }
}