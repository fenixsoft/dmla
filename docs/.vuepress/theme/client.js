import { defineClientConfig } from 'vuepress/client'
import { setupHeaders, setupSidebarItems } from '@theme/useSidebarItems'
import { setupDarkMode } from '@theme/useDarkMode'
import Layout from './layouts/Layout.vue'
import Swiper from '../components/Swiper.vue'
import Slide from '../components/Slide.vue'
import './styles/index.scss'

export default defineClientConfig({
  layouts: {
    Layout,
  },
  enhanceApp({ app }) {
    // 注册全局组件
    app.component('Swiper', Swiper)
    app.component('Slide', Slide)
  },
  setup() {
    // 初始化默认主题的功能
    setupDarkMode()
    setupHeaders()
    setupSidebarItems()
  }
})