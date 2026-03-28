<template>
  <header class="navbar">
    <div class="navbar-container">
      <!-- Logo -->
      <RouterLink to="/" class="navbar-brand">
        <span class="brand-icon"></span>
        <span class="brand-text">IdeaSpaces</span>
      </RouterLink>

      <!-- 导航链接 -->
      <nav class="navbar-links">
        <RouterLink
          v-for="item in navbarItems"
          :key="item.link"
          :to="item.link"
          class="nav-link"
          :class="{ active: isActive(item.link) }"
        >
          {{ item.text }}
        </RouterLink>
      </nav>

      <!-- 占位符 -->
      <div class="navbar-spacer"></div>

      <!-- 设置按钮 -->
      <button class="settings-btn" @click="showSettings = true" title="沙箱设置">
        <svg class="settings-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
      </button>

      <!-- GitHub 按钮 -->
      <a
        :href="repoUrl"
        target="_blank"
        rel="noopener noreferrer"
        class="github-btn"
      >
        <svg class="github-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        <span>GitHub</span>
      </a>
    </div>

    <!-- 设置弹窗 -->
    <Settings :visible="showSettings" @close="showSettings = false" @save="onSettingsSave" />
  </header>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import { useThemeData } from '@vuepress/theme-default/lib/client/composables/index.js'
import Settings from './Settings.vue'

const route = useRoute()
const themeData = useThemeData()

// 设置弹窗状态
const showSettings = ref(false)

// 从主题配置中读取导航栏项目
const navbarItems = computed(() => {
  return themeData.value.navbar || []
})

// GitHub 仓库
const repoUrl = 'https://github.com/username/ideaspaces'

// 是否激活
function isActive(link) {
  if (link === '/') {
    return route.path === '/'
  }
  return route.path.startsWith(link)
}

// 设置保存回调
function onSettingsSave(config) {
  console.log('沙箱设置已保存:', config)
}
</script>

<style lang="scss" scoped>
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 64px;
  background: #FFFFFF;
  border-bottom: 1.5px solid #E4E4E7;
  z-index: 100;
}

.navbar-container {
  max-width: 1440px;
  height: 100%;
  margin: 0 auto;
  padding: 0 48px;
  display: flex;
  align-items: center;
}

.navbar-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  color: #18181B;
}

.brand-icon {
  width: 32px;
  height: 32px;
  background: #2563EB;
  border-radius: 8px;
}

.brand-text {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 18px;
  font-weight: 600;
  color: #18181B;
}

.navbar-links {
  display: flex;
  align-items: center;
  gap: 32px;
  margin-left: 48px;
}

.nav-link {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 14px;
  font-weight: 500;
  color: #71717A;
  text-decoration: none;
  transition: color 0.2s ease;

  &:hover {
    color: #18181B;
  }

  &.active {
    color: #2563EB;
  }
}

.navbar-spacer {
  flex: 1;
}

.github-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: 1.5px solid #E4E4E7;
  border-radius: 8px;
  color: #18181B;
  text-decoration: none;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;

  &:hover {
    background: #FAFAFA;
    border-color: #D4D4D8;
  }
}

.github-icon {
  width: 16px;
  height: 16px;
  color: #71717A;
}

.settings-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  padding: 0;
  border: 1.5px solid #E4E4E7;
  border-radius: 8px;
  background: #fff;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-right: 8px;
}

.settings-btn:hover {
  background: #FAFAFA;
  border-color: #D4D4D8;
}

.settings-icon {
  width: 18px;
  height: 18px;
  color: #71717A;
}
</style>