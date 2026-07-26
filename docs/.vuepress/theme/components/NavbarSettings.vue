<template>
  <div class="navbar-settings">
    <!-- GitHub 链接 -->
    <a
      class="github-link"
      href="https://github.com/fenixsoft/dmla"
      target="_blank"
      rel="noopener noreferrer"
      title="GitHub 仓库"
    >
      <svg class="github-icon" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.835 1.305 3.51.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.32.4725-2.4 1.23-3.24-.12-.3-.54-1.515.12-3.15 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.635.24 2.85.12 3.15.765.84 1.23 1.905 1.23 3.24 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
      </svg>
    </a>

    <!-- 语言切换按钮 -->
    <button class="locale-btn" @click="switchLocale" :title="isEnglish ? '切换到中文' : 'Switch to English'">
      <svg class="locale-icon" viewBox="0 0 24 24" fill="none">
        <!-- 圆形边框 -->
        <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="1.5"/>
        <template v-if="isEnglish">
          <!-- 中字：竖笔 + 口（描边，可见空隙） -->
          <line x1="12" y1="6" x2="12" y2="18" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          <rect x="6" y="8.5" width="12" height="6" rx="1" fill="none" stroke="currentColor" stroke-width="1.5"/>
        </template>
        <template v-else>
          <text x="12" y="16" font-size="10" font-weight="700" fill="currentColor" font-family="Arial, sans-serif" text-anchor="middle">EN</text>
        </template>
      </svg>
    </button>

    <!-- 设置按钮 -->
    <button class="settings-btn" @click="showSettings = true" title="设置">
      <svg class="settings-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
      </svg>
    </button>

    <!-- 设置弹窗 -->
    <Settings :visible="showSettings" @close="showSettings = false" @save="onSettingsSave" />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute } from '@vuepress/client'
import Settings from './Settings.vue'

const route = useRoute()

// 设置弹窗状态
const showSettings = ref(false)

// 当前是否在英文页面
const isEnglish = computed(() => route.path.startsWith('/en/'))

// 切换语言：中→英、英→中
function switchLocale() {
  const currentPath = route.path
  let targetPath

  if (isEnglish.value) {
    // 英文 → 中文：去掉 /en/ 前缀
    targetPath = currentPath.replace(/^\/en/, '') || '/'
  } else {
    // 中文 → 英文：在路径前加 /en/
    targetPath = '/en' + currentPath
    targetPath = targetPath.replace(/\/+/g, '/')
  }

  if (targetPath !== currentPath) {
    window.location.href = targetPath
  }
}

// 设置保存回调
function onSettingsSave(config) {
  console.log('沙箱设置已保存:', config)
}
</script>

<style scoped>
.navbar-settings {
  display: flex;
  align-items: center;
}

.github-link {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  margin-left: 12px;
  border-radius: 4px;
  background: transparent;
  transition: background-color 0.2s ease;
}

.github-link:hover {
  background: var(--vp-c-control-hover);
}

.github-icon {
  width: 18px;
  height: 18px;
  color: var(--vp-c-text);
}

.settings-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  margin-left: 12px;
  padding: 0;
  border: none;
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.settings-btn:hover {
  background: var(--vp-c-control-hover);
}

.settings-icon {
  width: 18px;
  height: 18px;
  color: var(--vp-c-text);
}

.locale-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  margin-left: 12px;
  padding: 0;
  border: none;
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.locale-btn:hover {
  background: var(--vp-c-control-hover);
}

.locale-icon {
  width: 22px;
  height: 22px;
  color: var(--vp-c-text);
}

/* 移动端：缩小按钮间距和尺寸 */
@media (max-width: 719px) {
  .github-link,
  .settings-btn,
  .locale-btn {
    margin-left: 6px;
  }
}

@media (max-width: 419px) {
  .github-link,
  .settings-btn,
  .locale-btn {
    width: 24px;
    height: 24px;
    margin-left: 4px;
  }

  .github-icon,
  .settings-icon {
    width: 16px;
    height: 16px;
  }

  .locale-icon {
    width: 18px;
    height: 18px;
  }
}
</style>