/**
 * Issue 同步脚本
 * 自动为新文章创建 GitHub Issue，并回写编号到文章 frontmatter
 */

import { readFileSync, readdirSync, statSync, writeFileSync } from 'fs'
import { join } from 'path'

// 配置
const DOCS_DIR = './docs'
// 使用 Gitalk 的标签格式，确保 Gitalk 能通过标签找到 issue
// Gitalk 查找逻辑：labels.concat(id)，即 ['Gitalk', 'Comment', pathname]
const ISSUE_LABELS = ['Gitalk', 'Comment']

/**
 * 解析 Markdown frontmatter
 */
function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) return {}

  const frontmatter = {}
  const lines = match[1].split('\n')

  let currentKey = null
  let currentObj = null

  for (const line of lines) {
    // 简单的 YAML 解析
    if (line.startsWith('  ')) {
      // 嵌套属性
      if (currentObj) {
        const [key, value] = line.trim().split(':').map(s => s.trim())
        if (value) {
          currentObj[key] = parseValue(value)
        }
      }
    } else if (line.includes(':')) {
      const [key, value] = line.split(':').map(s => s.trim())
      if (key === 'issue') {
        currentKey = 'issue'
        currentObj = {}
        frontmatter.issue = currentObj
      } else if (value) {
        frontmatter[key] = parseValue(value)
        currentObj = null
      }
    }
  }

  return frontmatter
}

/**
 * 解析 YAML 值
 */
function parseValue(value) {
  if (value.startsWith('"') && value.endsWith('"')) {
    return value.slice(1, -1)
  }
  if (value.startsWith('[') && value.endsWith(']')) {
    return value.slice(1, -1).split(',').map(s => s.trim())
  }
  if (value === 'true') return true
  if (value === 'false') return false
  if (!isNaN(value)) return Number(value)
  return value
}

/**
 * 更新 frontmatter，添加 issue.number 字段
 */
function updateFrontmatterWithIssueNumber(filePath, issueNumber) {
  const content = readFileSync(filePath, 'utf-8')

  // 匹配 frontmatter
  const match = content.match(/^(---\n[\s\S]*?\n---)/)
  if (!match) {
    console.log(`  ⚠️  无法解析 frontmatter，跳过回写`)
    return false
  }

  const frontmatterBlock = match[1]

  // 检查是否已有 issue 字段
  if (frontmatterBlock.includes('issue:')) {
    // 检查是否已有 number 字段
    if (/issue:\s*\n\s*number:/.test(frontmatterBlock)) {
      // 替换已有的 number
      const updatedFrontmatter = frontmatterBlock.replace(
        /(issue:\s*\n\s*number:\s*)\d+/,
        `$1${issueNumber}`
      )
      const newContent = content.replace(frontmatterBlock, updatedFrontmatter)
      writeFileSync(filePath, newContent, 'utf-8')
    } else {
      // 在 issue: 后添加 number
      const updatedFrontmatter = frontmatterBlock.replace(
        /(issue:\s*\n)/,
        `$1  number: ${issueNumber}\n`
      )
      const newContent = content.replace(frontmatterBlock, updatedFrontmatter)
      writeFileSync(filePath, newContent, 'utf-8')
    }
  } else {
    // 没有 issue 字段，在 frontmatter 结尾添加
    const updatedFrontmatter = frontmatterBlock.replace(
      /\n---$/,
      `\nissue:\n  number: ${issueNumber}\n---`
    )
    const newContent = content.replace(frontmatterBlock, updatedFrontmatter)
    writeFileSync(filePath, newContent, 'utf-8')
  }

  return true
}

/**
 * 递归获取所有 Markdown 文件
 */
function getMarkdownFiles(dir, files = []) {
  const items = readdirSync(dir)

  for (const item of items) {
    const fullPath = join(dir, item)
    const stat = statSync(fullPath)

    if (stat.isDirectory()) {
      // 跳过 .vuepress 目录
      if (item !== '.vuepress') {
        getMarkdownFiles(fullPath, files)
      }
    } else if (item.endsWith('.md')) {
      files.push(fullPath)
    }
  }

  return files
}

/**
 * 将文件路径转换为 Gitalk 使用的 pathname (id)
 * 例如: ./docs/python/decorators.md -> /python/decorators.html
 */
function filePathToPathname(filePath) {
  // 移除 ./docs 前缀和 .md 后缀，添加 .html
  const relativePath = filePath
    .replace(/^\.\//, '')
    .replace(/^docs\//, '')
    .replace(/\.md$/, '.html')
  return '/' + relativePath
}

/**
 * 创建 GitHub Issue
 * @param {string} title - Issue 标题
 * @param {string[]} labels - 基础标签
 * @param {string} pathname - 文章路径（作为 Gitalk id 标签）
 */
async function createIssue(title, labels, pathname) {
  const token = process.env.GITHUB_TOKEN
  const repo = process.env.GITHUB_REPOSITORY

  if (!token || !repo) {
    console.log('⚠️  GitHub Token 或 Repository 未配置')
    return null
  }

  const [owner, repoName] = repo.split('/')

  // Gitalk 查找 issue 的方式是 labels.concat(id)
  // 所以我们需要把 pathname 作为 id 标签加入
  const allLabels = labels.concat(pathname)

  const body = `# 讨论

这是文章 "${title}" 的评论区。

欢迎在这里留言讨论！`

  const response = await fetch(`https://api.github.com/repos/${owner}/${repoName}/issues`, {
    method: 'POST',
    headers: {
      'Authorization': `token ${token}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      title,
      body,
      labels: allLabels
    })
  })

  if (!response.ok) {
    throw new Error(`Failed to create issue: ${response.status}`)
  }

  return await response.json()
}

/**
 * 搜索已存在的 Issue（使用 Gitalk 的标签组合方式）
 * @param {string[]} labels - 基础标签
 * @param {string} pathname - 文章路径（作为 Gitalk id 标签）
 */
async function findIssue(labels, pathname) {
  const token = process.env.GITHUB_TOKEN
  const repo = process.env.GITHUB_REPOSITORY

  if (!token || !repo) return null

  const [owner, repoName] = repo.split('/')

  // Gitalk 查找 issue 的方式：labels.concat(id).join(',')
  const allLabels = labels.concat(pathname).join(',')
  const query = encodeURIComponent(`repo:${owner}/${repoName} is:issue ${allLabels}`)
  const response = await fetch(`https://api.github.com/search/issues?q=${query}`, {
    headers: {
      'Authorization': `token ${token}`,
      'Accept': 'application/vnd.github.v3+json'
    }
  })

  if (!response.ok) return null

  const data = await response.json()
  return data.items[0]
}

/**
 * 主函数
 */
async function main() {
  console.log('🔄 开始同步 Issues...\n')

  const files = getMarkdownFiles(DOCS_DIR)
  console.log(`找到 ${files.length} 个 Markdown 文件\n`)

  for (const file of files) {
    // 跳过首页和目录索引
    if (file.endsWith('README.md') || file.endsWith('design.md')) {
      continue
    }

    const content = readFileSync(file, 'utf-8')
    const frontmatter = parseFrontmatter(content)

    // 跳过没有标题的文件
    if (!frontmatter.title) continue

    // 计算文章 pathname（Gitalk 用这个作为 id 标签）
    const pathname = filePathToPathname(file)

    // 确定 Issue 标题（直接使用文章标题，不加前缀）
    const issueTitle = frontmatter.issue?.title || frontmatter.title

    // 检查是否已有 Issue 编号
    if (frontmatter.issue?.number) {
      console.log(`✓ ${frontmatter.title} -> Issue #${frontmatter.issue.number} (已存在)`)
      continue
    }

    try {
      // 搜索现有 Issue（使用 Gitalk 的标签组合方式）
      const existingIssue = await findIssue(ISSUE_LABELS, pathname)

      if (existingIssue) {
        console.log(`✓ ${frontmatter.title} -> Issue #${existingIssue.number} (已存在，回写编号)`)
        // 回写编号到 frontmatter
        const updated = updateFrontmatterWithIssueNumber(file, existingIssue.number)
        if (updated) {
          console.log(`  ✓ 已更新 frontmatter: issue.number = ${existingIssue.number}`)
        }
        continue
      }

      // 创建新 Issue（使用 Gitalk 的标签格式）
      const newIssue = await createIssue(issueTitle, ISSUE_LABELS, pathname)

      if (newIssue) {
        console.log(`✓ ${frontmatter.title} -> Issue #${newIssue.number} (已创建)`)
        console.log(`  标签: ${ISSUE_LABELS.join(', ')}, ${pathname}`)

        // 回写编号到 frontmatter
        const updated = updateFrontmatterWithIssueNumber(file, newIssue.number)
        if (updated) {
          console.log(`  ✓ 已更新 frontmatter: issue.number = ${newIssue.number}`)
        }
      }

    } catch (error) {
      console.log(`✗ ${frontmatter.title} -> ${error.message}`)
    }
  }

  console.log('\n✅ Issue 同步完成')
}

main().catch(console.error)