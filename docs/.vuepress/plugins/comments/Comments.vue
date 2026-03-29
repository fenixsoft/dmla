<template>
  <div class="comments-section" ref="commentsContainer">
    <div id="gitalk-container"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import Gitalk from 'gitalk'
import 'gitalk/dist/gitalk.css'

const route = useRoute()
const commentsContainer = ref(null)

// Gitalk 配置
const gitalkConfig = {
  clientID: 'Ov23liQbtfoZGMcsU9VV',
  repo: 'ideaspaces',
  owner: 'fenixsoft',
  admin: ['fenixsoft'],
  id: location.pathname,
  title: document.title,
  body: location.href,
  labels: ['Gitalk', 'Comment'],
  distractionFreeMode: false,
  proxy: 'http://cros.icyfenix.cn/callback?code={code}',
  createIssueManually: false,
  pagerDirection: 'last',
  enableHotKey: true
}

// 初始化 Gitalk
const initGitalk = () => {
  const gitalk = new Gitalk(gitalkConfig)
  gitalk.render('gitalk-container')
}

onMounted(() => {
  initGitalk()
})

// 路由变化时重新渲染
watch(() => route.path, () => {
  const container = document.getElementById('gitalk-container')
  if (container) {
    container.innerHTML = ''
  }
  initGitalk()
})
</script>

<style scoped>
.comments-section {
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid var(--c-border, #eaecef);
}
</style>