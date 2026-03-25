/**
 * 评论插件客户端配置
 */
import { defineClientConfig } from 'vuepress/client'
import Comments from './Comments.vue'

export default defineClientConfig({
  enhance({ app }) {
    app.component('Comments', Comments)
  }
})