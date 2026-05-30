/**
 * Mermaid 插件 - 支持 Markdown 中的流程图
 */
import { getDirname, path } from 'vuepress/utils'

const __dirname = getDirname(import.meta.url)

export default {
  name: 'vuepress-plugin-mermaid',

  extendsMarkdown: (md) => {
    const defaultFence = md.renderer.rules.fence

    md.renderer.rules.fence = (tokens, idx, options, env, self) => {
      const token = tokens[idx]
      const code = token.content.trim()
      const lang = token.info.trim().toLowerCase()

      // 支持大小修饰符：mermaid small, mermaid compact, mermaid tiny
      // small: zoom 0.85, compact: zoom 0.75, tiny: zoom 0.6
      if (lang.startsWith('mermaid')) {
        // 解析大小修饰符，只接受 small, compact, tiny
        const parts = lang.split(/\s+/)
        const sizeModifier = parts[1]
        const validSizes = ['small', 'compact', 'tiny']
        const sizeClass = validSizes.includes(sizeModifier) ? `mermaid-${sizeModifier}` : ''

      // Pre-process: move Chinese text outside $$...$$ to avoid KaTeX warnings
      let fixedCode = code;
      fixedCode = fixedCode.replace(/\$\$([^\$]*?)\$\$/g, (match, inner) => {
        if (!/[\u4e00-\u9fff]/.test(inner)) return match;
        const m = inner.match(/^([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\uff0c\u3002\uff1a\uff1b\uff01\uff1f\u3001\s]+)(.*)$/);
        if (m) {
          const chinese = m[1].trimEnd();
          const math = m[2].trimStart();
          if (math && !/[\u4e00-\u9fff]/.test(math)) {
            return chinese + ' $$' + math + '$$';
          }
        }
        const cleaned = inner.replace(/[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\uff0c\u3002\uff1a\uff1b\uff01\uff1f\u3001]/g, '').trim();
        return cleaned ? '$$' + cleaned + '$$' : inner;
      });

        // 使用 data-mermaid 属性存储原始代码，避免浏览器解析 HTML 标签
        // encodeURIComponent 确保 <br> 等标签不会被浏览器规范化
        return `<pre class="mermaid ${sizeClass}" data-mermaid="${encodeURIComponent(fixedCode)}"></pre>`
      }

      return defaultFence(tokens, idx, options, env, self)
    }
  },

  clientConfigFile: path.resolve(__dirname, 'client.js')
}
