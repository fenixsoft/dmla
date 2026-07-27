<template>
  <span class="github-infos">
    <GithubButton
      :href="`https://github.com/${repo}`"
      data-icon="octicon-star"
      data-show-count="true"
      :data-text="'Star'"
      :aria-label="`Star ${repo} on GitHub`"
    />
    <GithubButton
      :href="`https://github.com/${owner}`"
      data-show-count="true"
      :data-text="'Follow'"
      :aria-label="`Follow @${owner} on GitHub`"
    />
    <GithubButton
      :href="`https://github.com/${repo}/fork`"
      data-icon="octicon-repo-forked"
      data-show-count="true"
      :data-text="'Fork'"
      :aria-label="`Fork ${repo} on GitHub`"
    />
  </span>
</template>

<script setup>
import GithubButton from './GithubButton.vue'

defineProps({
  repo: {
    type: String,
    default: 'fenixsoft/dmla'
  },
  owner: {
    type: String,
    default: 'fenixsoft'
  }
})
</script>

<style scoped>
.github-infos {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
  justify-content: center;
}
</style>
