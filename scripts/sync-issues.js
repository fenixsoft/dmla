/**
 * Issue 同步脚本
 * 自动为新文章创建 GitHub Issue
 */

import { readFileSync, readdirSync, statSync } from 'fs'
import { join, relative } from 'path'
import { execSync } from 'child_process'

// 配置
const DOCS_DIR = './docs'
const ISSUE_LABELS = ['comments', 'article']
const ISSUE_PREFIX = '[Comments] '

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
 * 创建 GitHub Issue
 */
async function createIssue(title, labels) {
  const token = process.env.GITHUB_TOKEN
  const repo = process.env.GITHUB_REPOSITORY

  if (!token || !repo) {
    console.log('⚠️  GitHub Token 或 Repository 未配置')
    return null
  }

  const [owner, repoName] = repo.split('/')

  const body = `# 讨论

这是文章 "${title.replace(ISSUE_PREFIX, '')}" 的评论区。

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
      labels
    })
  })

  if (!response.ok) {
    throw new Error(`Failed to create issue: ${response.status}`)
  }

  return await response.json()
}

/**
 * 搜索现有 Issue
 */
async function findIssue(title) {
  const token = process.env.GITHUB_TOKEN
  const repo = process.env.GITHUB_REPOSITORY

  if (!token || !repo) return null

  const [owner, repoName] = repo.split('/')

  const query = encodeURIComponent(`${title} repo:${owner}/${repoName} is:issue label:comments`)
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

    // 确定 Issue 标题
    const issueTitle = frontmatter.issue?.title
      ? `${ISSUE_PREFIX}${frontmatter.issue.title}`
      : `${ISSUE_PREFIX}${frontmatter.title}`

    // 检查是否已有 Issue 编号
    if (frontmatter.issue?.number) {
      console.log(`✓ ${frontmatter.title} -> Issue #${frontmatter.issue.number} (已存在)`)
      continue
    }

    try {
      // 搜索现有 Issue
      const existingIssue = await findIssue(issueTitle)

      if (existingIssue) {
        console.log(`✓ ${frontmatter.title} -> Issue #${existingIssue.number} (已存在)`)
        continue
      }

      // 创建新 Issue
      const newIssue = await createIssue(issueTitle, ISSUE_LABELS)

      if (newIssue) {
        console.log(`✓ ${frontmatter.title} -> Issue #${newIssue.number} (已创建)`)
      }

    } catch (error) {
      console.log(`✗ ${frontmatter.title} -> ${error.message}`)
    }
  }

  console.log('\n✅ Issue 同步完成')
}

main().catch(console.error)