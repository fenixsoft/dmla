/**
 * 修复 markdown-it 在处理中文括号后粗体标记的问题
 *
 * 问题：根据 CommonMark 规范，当 `**` 前面是中文括号 `）`（标点），
 * 后面是普通文字时，can_close=false，导致无法闭合粗体。
 *
 * 解决方案：在渲染前用正则表达式预处理，将包含中文右括号的粗体
 * 内容替换为 HTML <strong> 标签。
 */
import { path } from 'vuepress/utils'

const __dirname = import.meta.dirname || path.dirname(new URL(import.meta.url).pathname)

/**
 * 匹配 **...）** 模式
 * 其中内容不包含 ** 且以中文右括号或英文右括号结尾
 */
const BOLD_FIX_PATTERN = /\*\*([^*]*[）)])\*\*/g

export const emphasisFixPlugin = {
  name: 'vuepress-plugin-emphasis-fix',

  extendsMarkdown: (md) => {
    // 保存原始的 parse 方法
    const originalParse = md.parse.bind(md)

    // 重写 parse 方法，在解析前预处理源码
    md.parse = function(src, env) {
      // 预处理：将 **...）** 替换为 <strong>...</strong>
      // 这样可以绕过 markdown-it 的 emphasis 解析问题
      const fixedSrc = src.replace(BOLD_FIX_PATTERN, '<strong>$1</strong>')

      return originalParse(fixedSrc, env)
    }
  }
}

export default emphasisFixPlugin