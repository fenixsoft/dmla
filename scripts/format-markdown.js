#!/usr/bin/env node
/**
 * Markdown 格式化脚本
 *
 * 功能：
 * 1. 中英文之间添加空格（汉字与半角字母数字）
 * 2. 中文与运算符符号之间添加空格（= + - * / > < 等）
 * 3. 全角双引号替换为半角引号
 * 4. 加粗格式统一（**中文（English）** → **中文**（English））
 * 5. 清理中文标点周围的空格
 *
 * 用法：
 *   npm run format:md        # 检查模式（默认）
 *   npm run format:md:check  # 检查模式
 *   npm run format:md:fix    # 修复模式
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 项目根目录
const PROJECT_ROOT = path.resolve(__dirname, '..');
const DOCS_DIR = path.join(PROJECT_ROOT, 'docs');

// 解析命令行参数
const args = process.argv.slice(2);
const mode = args.includes('--fix') ? 'fix' : 'check';

// 统计信息
const stats = {
  filesScanned: 0,
  filesModified: 0,
  totalIssues: 0
};

// 占位符分隔符 - 使用 \x01 (SOH) 确保不会被任何正则表达式匹配跨越
const PLACEHOLDER_DELIMITER = '\x01';

/**
 * 保护排除区域，用占位符替换
 * @param {string} content 文件内容
 * @returns {Object} { protectedContent, placeholders }
 */
function protectExcludedRegions(content) {
  const placeholders = {
    mathBlock: [],
    mathInline: [],
    codeBlock: [],
    codeInline: [],
    frontmatter: [],
    linkUrl: []
  };

  let protectedContent = content;

  // 1. 保护 frontmatter (---...---)
  const frontmatterRegex = /^---\n[\s\S]*?\n---/;
  protectedContent = protectedContent.replace(frontmatterRegex, (match) => {
    const index = placeholders.frontmatter.length;
    placeholders.frontmatter.push(match);
    return `{{${PLACEHOLDER_DELIMITER}FRONTMATTER_${index}${PLACEHOLDER_DELIMITER}}}`;
  });

  // 2. 保护 Markdown 链接 URL 部分（括号内的内容）
  // 格式：[text](url)，其中 url 不应该被格式化（包含锚点等）
  // 注意：只保护 URL 部分，不保护链接文本
  const linkUrlRegex = /\[[^\[\]]+\]\([^\x01()]+\)/g;
  protectedContent = protectedContent.replace(linkUrlRegex, (match) => {
    // 分离文本和 URL
    const textMatch = match.match(/\[([^\[\]]+)\]\(([^)]+)\)/);
    if (textMatch) {
      const text = textMatch[1];
      const url = textMatch[2];
      const index = placeholders.linkUrl.length;
      placeholders.linkUrl.push(url);
      // 只替换 URL 部分，保留文本
      return `[${text}]({{${PLACEHOLDER_DELIMITER}LINK_URL_${index}${PLACEHOLDER_DELIMITER}}})`;
    }
    return match;
  });

  // 3. 保护公式块 ($$...$$)
  // 使用 [^\x01] 确保不会跨越已保护的区域
  // 使用 +? 非贪婪匹配，避免匹配多个公式块时跨越中间内容
  const mathBlockRegex = /\$\$[\s\S]+?\$\$/g;
  protectedContent = protectedContent.replace(mathBlockRegex, (match) => {
    const index = placeholders.mathBlock.length;
    placeholders.mathBlock.push(match);
    return `{{${PLACEHOLDER_DELIMITER}MATH_BLOCK_${index}${PLACEHOLDER_DELIMITER}}}`;
  });

  // 4. 保护代码块 (```...```)
  const codeBlockRegex = /```[\s\S]+?```/g;
  protectedContent = protectedContent.replace(codeBlockRegex, (match) => {
    const index = placeholders.codeBlock.length;
    placeholders.codeBlock.push(match);
    return `{{${PLACEHOLDER_DELIMITER}CODE_BLOCK_${index}${PLACEHOLDER_DELIMITER}}}`;
  });

  // 5. 保护行内代码 (`...`)
  // 使用 [^\x01`] 确保不会跨越已保护的区域（占位符包含 \x01）
  const codeInlineRegex = /`[^\x01`]+`/g;
  protectedContent = protectedContent.replace(codeInlineRegex, (match) => {
    const index = placeholders.codeInline.length;
    placeholders.codeInline.push(match);
    return `{{${PLACEHOLDER_DELIMITER}CODE_INLINE_${index}${PLACEHOLDER_DELIMITER}}}`;
  });

  // 6. 保护行内公式 ($...$)
  // 使用 [^\x01$\n] 确保不会跨越已保护的区域，也不包含换行
  const mathInlineRegex = /\$[^\x01$\n]+\$/g;
  protectedContent = protectedContent.replace(mathInlineRegex, (match) => {
    const index = placeholders.mathInline.length;
    placeholders.mathInline.push(match);
    return `{{${PLACEHOLDER_DELIMITER}MATH_INLINE_${index}${PLACEHOLDER_DELIMITER}}}`;
  });

  return { protectedContent, placeholders };
}

/**
 * 还原排除区域
 * @param {string} content 带占位符的内容
 * @param {Object} placeholders 占位符映射
 * @returns {string} 还原后的内容
 */
function restoreExcludedRegions(content, placeholders) {
  let restored = content;

  // camelCase 转换为下划线格式
  const typeToPrefix = (type) => {
    return type.replace(/([a-z])([A-Z])/g, '$1_$2').toUpperCase();
  };

  // 按顺序还原各类型占位符
  // 注意：需要转义替换字符串中的 $ 字符，因为 $$ 在 replace 中有特殊含义
  for (const [type, items] of Object.entries(placeholders)) {
    const prefix = typeToPrefix(type);
    for (let i = 0; i < items.length; i++) {
      // 新格式：{{\x01PREFIX_N\x01}}
      const placeholder = `{{${PLACEHOLDER_DELIMITER}${prefix}_${i}${PLACEHOLDER_DELIMITER}}}`;
      // 将 $ 转义为 $$（在替换字符串中 $$ 表示字面量 $）
      const escapedItem = items[i].replace(/\$/g, '$$$$');
      restored = restored.replace(placeholder, escapedItem);
    }
  }

  return restored;
}

/**
 * 清理中文标点周围的空格
 * 中文标点符号与汉字之间不应该有空格（但换行符是正常的段落分隔，不应处理）
 * @param {string} content 内容
 * @returns {string} 处理后的内容
 */
function cleanPunctuationSpacing(content) {
  // 中文全角标点符号范围：
  // \u3000-\u303F: CJK符号和标点（如、，。！？：；）
  // \uFF00-\uFFEF: 全角ASCII变体（如全角括号（）【】）

  // 注意：只处理行内的空格，不处理包含换行符的情况
  // 使用 [ \t]+ 只匹配空格和制表符，排除 \n \r

  // 全角标点后有空格/制表符再接汉字 → 去掉空格
  content = content.replace(/([\u3000-\u303F\uFF08\uFF09\uFF3B\uFF3D])([ \t]+)([\u4e00-\u9fff])/g, (match, punct, space, chinese) => {
    return `${punct}${chinese}`;
  });

  // 汉字后有空格/制表符再接全角标点 → 去掉空格
  content = content.replace(/([\u4e00-\u9fff])([ \t]+)([\u3000-\u303F\uFF08\uFF09\uFF3B\uFF3D])/g, (match, chinese, space, punct) => {
    return `${chinese}${punct}`;
  });

  return content;
}

/**
 * 中英文添加空格
 * @param {string} content 内容
 * @returns {string} 处理后的内容
 */
function addSpaceBetweenChineseAndEnglish(content) {
  // 中文汉字范围：\u4e00-\u9fff (不含标点)
  // 半角字符：英文字母 A-Z, a-z 和数字 0-9

  // 汉字后接半角字符，中间无空格 → 添加空格
  content = content.replace(/([\u4e00-\u9fff])([A-Za-z0-9])/g, (match, chinese, english) => {
    return `${chinese} ${english}`;
  });

  // 半角字符后接汉字，中间无空格 → 添加空格
  content = content.replace(/([A-Za-z0-9])([\u4e00-\u9fff])/g, (match, english, chinese) => {
    return `${english} ${chinese}`;
  });

  return content;
}

/**
 * 中文与运算符符号添加空格
 * @param {string} content 内容
 * @returns {string} 处理后的内容
 */
function addSpaceBetweenChineseAndOperator(content) {
  // 运算符符号：= + - > < ——（全角破折号）
  // 规则：运算符前后都必须是中文汉字（排除 HTML 标签、Markdown 标记等）
  // 注意：
  // - 排除 *（Markdown 加粗标记）和 _（Markdown 斜体标记）
  // - 排除 /（常用于路径、分数，不作为运算符处理）
  // - 包含 ——（全角破折号，中文语境中常作为连接符号）

  // 中文汉字后接运算符再接中文汉字 → 添加空格（运算符前后）
  // 例如：天气=雨天 → 天气 = 雨天
  content = content.replace(/([\u4e00-\u9fff])([=+\-])([\u4e00-\u9fff])/g, (match, before, op, after) => {
    return `${before} ${op} ${after}`;
  });

  // 比较运算符单独处理（> 和 < 容易与 HTML 标签混淆）
  // 只处理：中文>中文 或 中文<中文（前后都是中文）
  content = content.replace(/([\u4e00-\u9fff])([><])([\u4e00-\u9fff])/g, (match, before, op, after) => {
    return `${before} ${op} ${after}`;
  });

  // 全角破折号（——）单独处理
  // 例如：机器学习——深度学习 → 机器学习 —— 深度学习
  // 扩展：允许破折号前后是中文汉字、引号、Markdown标记（* 和 _）
  // 处理情况如："标准件"——虽然、两类问题——**求切线**
  content = content.replace(/([\u4e00-\u9fff"'*_])(——)([\u4e00-\u9fff"'*_])/g, (match, before, dash, after) => {
    return `${before} ${dash} ${after}`;
  });

  return content;
}

/**
 * 全角引号替换为半角引号
 * @param {string} content 内容
 * @returns {string} 处理后的内容
 */
function replaceFullWidthQuotes(content) {
  // 全角双引号 " 和 " 替换为半角 "
  content = content.replace(/"/g, '"');
  content = content.replace(/"/g, '"');
  return content;
}

/**
 * 加粗格式统一
 * 将 "**中文概念（English Name）**" 改为 "**中文概念**（English Name）"
 * @param {string} content 内容
 * @returns {string} 处理后的内容
 */
function unifyBoldFormat(content) {
  // 匹配 **中文内容（英文内容）** 格式
  // 需要区分全角括号（）和半角括号()
  // 括号内内容应该是英文（包含字母）

  // 全角括号情况：**中文（English）** → **中文**（English）
  content = content.replace(/\*\*([^*]+)（([A-Za-z][^）]*)）\*\*/g, (match, chinese, english) => {
    // 检查 chinese 是否主要是中文（不包含公式占位符等）
    if (chinese.includes('{{') || chinese.includes('$')) {
      return match; // 包含特殊内容，不处理
    }
    return `**${chinese}**（${english}）`;
  });

  // 半角括号情况：**中文(English)** → **中文**(English)
  content = content.replace(/\*\*([^*]+)\(([A-Za-z][^\)]*)\)\*\*/g, (match, chinese, english) => {
    if (chinese.includes('{{') || chinese.includes('$')) {
      return match;
    }
    return `**${chinese}**(${english})`;
  });

  return content;
}

/**
 * 格式化单个文件内容
 * @param {string} content 文件内容
 * @returns {Object} { formatted: 格式化后的内容, issues: 发现的问题列表 }
 */
function formatContent(content, filePath) {
  const issues = [];

  // 1. 保护排除区域
  const { protectedContent: protectedText, placeholders } = protectExcludedRegions(content);

  // 2. 执行格式化
  let formatted = protectedText;

  // 2.1 中英文空格
  const beforeSpacing = formatted;
  formatted = addSpaceBetweenChineseAndEnglish(formatted);
  if (beforeSpacing !== formatted) {
    const spacingChanges = countChanges(beforeSpacing, formatted);
    if (spacingChanges > 0) {
      issues.push({ type: 'spacing', count: spacingChanges });
    }
  }

  // 2.2 中文与运算符空格
  const beforeOpSpacing = formatted;
  formatted = addSpaceBetweenChineseAndOperator(formatted);
  if (beforeOpSpacing !== formatted) {
    issues.push({ type: 'operator_spacing', count: 1 });
  }

  // 2.3 全角引号
  const beforeQuotes = formatted;
  formatted = replaceFullWidthQuotes(formatted);
  if (beforeQuotes !== formatted) {
    const quoteChanges = (beforeQuotes.match(/[""]/g) || []).length;
    if (quoteChanges > 0) {
      issues.push({ type: 'quotes', count: quoteChanges });
    }
  }

  // 2.4 加粗格式统一
  const beforeBold = formatted;
  formatted = unifyBoldFormat(formatted);
  if (beforeBold !== formatted) {
    issues.push({ type: 'bold', count: 1 });
  }

  // 2.5 清理中文标点周围的空格
  const beforePunct = formatted;
  formatted = cleanPunctuationSpacing(formatted);
  if (beforePunct !== formatted) {
    issues.push({ type: 'punctuation', count: 1 });
  }

  // 3. 还原排除区域
  formatted = restoreExcludedRegions(formatted, placeholders);

  return { formatted, issues };
}

/**
 * 计算变化数量（简化版本）
 */
function countChanges(before, after) {
  // 计算添加的空格数量
  const spacesBefore = (before.match(/[\u4e00-\u9fff] [A-Za-z0-9]|[A-Za-z0-9] [\u4e00-\u9fff]/g) || []).length;
  const spacesAfter = (after.match(/[\u4e00-\u9fff] [A-Za-z0-9]|[A-Za-z0-9] [\u4e00-\u9fff]/g) || []).length;
  return spacesAfter - spacesBefore;
}

/**
 * 递归获取目录下所有 .md 文件
 * @param {string} dir 目录路径
 * @param {string[]} excludeDirs 排除的目录
 * @returns {string[]} 文件路径列表
 */
function getMarkdownFiles(dir, excludeDirs = ['.vuepress']) {
  const files = [];

  function walk(currentDir) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        // 检查是否在排除列表中
        const relativePath = path.relative(dir, fullPath);
        if (!excludeDirs.some(ex => relativePath.startsWith(ex))) {
          walk(fullPath);
        }
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }
  }

  walk(dir);
  return files;
}

/**
 * 处理单个文件
 * @param {string} filePath 文件路径
 * @returns {Object} 处理结果
 */
function processFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const { formatted, issues } = formatContent(content, filePath);

  const hasChanges = content !== formatted;

  if (hasChanges && mode === 'fix') {
    fs.writeFileSync(filePath, formatted, 'utf-8');
  }

  return {
    filePath,
    relativePath: path.relative(PROJECT_ROOT, filePath),
    hasChanges,
    issues,
    original: content,
    formatted
  };
}

/**
 * 生成差异报告
 * @param {Object} result 处理结果
 * @returns {string} 报告文本
 */
function generateDiffReport(result) {
  const lines = [];
  lines.push(`\n  ${result.relativePath}`);

  if (result.issues.length > 0) {
    for (const issue of result.issues) {
      switch (issue.type) {
        case 'spacing':
          lines.push(`    - 中英文空格: ${issue.count} 处需要添加`);
          break;
        case 'operator_spacing':
          lines.push(`    - 运算符空格: 需要添加`);
          break;
        case 'quotes':
          lines.push(`    - 全角引号: ${issue.count} 处需要替换`);
          break;
        case 'bold':
          lines.push(`    - 加粗格式: 需要统一`);
          break;
        case 'punctuation':
          lines.push(`    - 标点空格: 需要清理`);
          break;
      }
    }
  }

  return lines.join('\n');
}

/**
 * 主函数
 */
function main() {
  console.log(`\nMarkdown 格式化脚本 (${mode === 'fix' ? '修复' : '检查'}模式)\n`);
  console.log(`扫描目录: ${DOCS_DIR}`);
  console.log(`排除目录: .vuepress/\n`);

  // 获取所有 Markdown 文件
  const files = getMarkdownFiles(DOCS_DIR, ['.vuepress']);
  stats.filesScanned = files.length;

  console.log(`找到 ${files.length} 个文件\n`);

  const results = [];

  // 处理每个文件
  for (const filePath of files) {
    const result = processFile(filePath);
    results.push(result);

    if (result.hasChanges) {
      stats.filesModified++;
      stats.totalIssues += result.issues.length;

      if (mode === 'check') {
        console.log(generateDiffReport(result));
      }
    }
  }

  // 输出汇总
  console.log('\n' + '='.repeat(50));
  console.log('\n汇总统计:');
  console.log(`  扫描文件: ${stats.filesScanned} 个`);
  console.log(`  需修改文件: ${stats.filesModified} 个`);
  console.log(`  问题总数: ${stats.totalIssues} 处`);

  if (mode === 'check' && stats.filesModified > 0) {
    console.log(`\n  运行 'npm run format:md:fix' 自动修复这些问题`);
  } else if (mode === 'fix') {
    console.log(`\n  ✓ 已完成 ${stats.filesModified} 个文件的格式化`);
  } else {
    console.log(`\n  ✓ 所有文件格式正确，无需修改`);
  }

  console.log('\n');
}

main();