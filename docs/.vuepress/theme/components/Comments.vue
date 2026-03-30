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

const props = defineProps({
  repo: {
    type: String,
    default: 'ideaspaces'
  },
  owner: {
    type: String,
    default: 'fenixsoft'
  },
  admin: {
    type: Array,
    default: () => ['fenixsoft']
  },
  clientId: {
    type: String,
    default: 'Ov23liQbtfoZGMcsU9VV'
  },
  proxy: {
    type: String,
    default: 'https://cros.icyfenix.cn/'
  }
})

// 生成 Gitalk 实例（只在客户端执行）
const initGitalk = () => {
  // 确保 title 不为空，使用 pathname 作为后备
  const pageTitle = document.title || location.pathname

  const gitalk = new Gitalk({
    clientID: props.clientId,
    // 不传 clientSecret，使用 proxy 代理
    repo: props.repo,
    owner: props.owner,
    admin: props.admin,
    id: location.pathname, // 页面的唯一标识，最长50字符
    title: pageTitle,
    body: location.href,
    labels: ['Gitalk', 'Comment'],
    distractionFreeMode: false,
    proxy: props.proxy,
    createIssueManually: false,
    pagerDirection: 'last',
    enableHotKey: true
  })

  gitalk.render('gitalk-container')
}

onMounted(() => {
  initGitalk()
})

// 路由变化时重新渲染
watch(() => route.path, () => {
  // 清空容器
  const container = document.getElementById('gitalk-container')
  if (container) {
    container.innerHTML = ''
  }
  initGitalk()
})
</script>

<style lang="scss" scoped>
.comments-section {
  margin-top: 32px;
  padding-top: 32px;
  border-top: 1.5px solid #E4E4E7;
}

:deep(.gt-container) {
  .gt-header {
    border-bottom: 1px solid #E4E4E7;
  }

  .gt-btn {
    background-color: #24292E;
    border-color: #24292E;

    &:hover {
      background-color: #32383F;
      border-color: #32383F;
    }
  }

  .gt-btn-preview {
    background-color: #fff;
    color: #24292E;
    border-color: #E4E4E7;

    &:hover {
      background-color: #F6F8FA;
    }
  }

  .gt-header-textarea {
    border-color: #E4E4E7;
    border-radius: 8px;

    &:focus {
      border-color: #2563EB;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
  }

  .gt-comment-content {
    border-radius: 8px;
    border: 1px solid #E4E4E7;
  }

  .gt-comment-body {
    color: #18181B;
  }
}
</style>