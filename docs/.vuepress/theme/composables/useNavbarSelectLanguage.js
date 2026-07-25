import { computed } from 'vue'

/**
 * 覆盖默认主题的 useNavbarSelectLanguage，返回空数组以禁用自动生成的语言下拉菜单。
 * 语言切换已通过自定义 NavbarSettings 组件中的图标实现。
 */
export const useNavbarSelectLanguage = () => {
  return computed(() => [])
}
