import { defineClientConfig } from 'vuepress/client'
import { setupHeaders, setupSidebarItems } from '@theme/useSidebarItems'
import { setupDarkMode } from '@theme/useDarkMode'
import Layout from './layouts/Layout.vue'
import './styles/index.scss'

export default defineClientConfig({
  layouts: {
    Layout,
  },
  setup() {
    // 初始化默认主题的功能
    setupDarkMode()
    setupHeaders()
    setupSidebarItems()
  },
})