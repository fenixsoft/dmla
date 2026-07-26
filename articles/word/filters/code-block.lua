-- articles/word/filters/code-block.lua
-- 代码块的 DOCX 输出样式说明
--
-- Pandoc DOCX writer 对代码块自动应用 "Source Code" 段落样式。
-- custom-style 属性仅在 Div / Span / Table 上受支持，对 CodeBlock 结点无效。
-- 如需自定义代码块样式，请在参考模板 (reference.docx) 中定义 "Source Code" 样式，
-- 或使用后处理脚本重命名 DOCX 中的 "Source Code" 段落样式。

function CodeBlock(el)
  -- 保留 Pandoc 默认行为：代码块自动应用 "Source Code" 样式。
  -- 视觉样式由参考模板中的 "Source Code" 定义控制。
  return el
end
