#!/usr/bin/env node
/**
 * 从文档中提取标记的类定义，同步到共享模块目录
 *
 * 用法: node scripts/extract-shared-modules.js
 *
 * 标记语法: ```python runnable extract-class="ClassName"
 * 多类语法: ```python runnable extract-class="ClassA, ClassB, func_name"
 *   逗号分隔的多个名称，同一代码块中的类和自由函数将合并写入同一个 .py 文件
 *   文件名以第一个名称为准
 *
 * 模块路径推断规则:
 *   1. 显式映射（CHAPTER_MAPPING）优先，用于特殊命名
 *   2. 未映射的目录自动推断：docs/<category>/<chapter>/ → shared/<chapter_snake_case>/
 *
 * 显式映射示例:
 *   'neural-network-structure' → 'neural' (简化命名)
 *   'convolutional-neural-network' → 'cnn' (简化命名)
 *
 * 自动推断示例:
 *   docs/deep-learning/transformer-models/ → shared/transformer_models/
 *   docs/statistical-learning/new-topic/ → shared/new_topic/
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 项目根目录
const PROJECT_ROOT = path.resolve(__dirname, '..');
const DOCS_DIR = path.join(PROJECT_ROOT, 'docs');
const SHARED_MODULES_DIR = path.join(PROJECT_ROOT, 'local-server', 'shared');

// 显式映射（用于特殊命名覆盖）
// 只在这里添加需要简化或特殊命名的目录
const CHAPTER_MAPPING = {
  // statistical-learning - 简化命名
  'statistical-learning/linear-models': 'linear',
  'statistical-learning/bayesian-methods': 'bayesian',
  'statistical-learning/support-vector-machines': 'svm',
  'statistical-learning/decision-tree-ensemble': 'tree',
  'statistical-learning/unsupervised-learning': 'unsupervised',
  // deep-learning - 简化命名
  'deep-learning/neural-network-structure': 'neural',
  'deep-learning/convolutional-neural-network': 'cnn',
  'deep-learning/generative-models': 'gan',
  // language-models - 简化命名
  'language-models/pretraining': 'llm'
};

// 将目录名转换为 snake_case 模块名
function dirToModuleName(dirName) {
  return dirName
    .replace(/-/g, '_')
    .toLowerCase();
}

// 推断模块路径（先查显式映射，再用自动推断）
function inferModulePath(docPath) {
  // 1. 查显式映射
  if (CHAPTER_MAPPING[docPath]) {
    return CHAPTER_MAPPING[docPath];
  }

  // 2. 自动推断：取最后一级目录名转为 snake_case
  // 例如: deep-learning/transformer-models → transformer_models
  const parts = docPath.split('/');
  const lastDir = parts[parts.length - 1];
  return dirToModuleName(lastDir);
}

// 常见技术术语（保持不拆分，作为完整单词）
const KNOWN_TERMS = [
  'ImageNet', 'AlexNet', 'VGGNet', 'ResNet', 'GoogleNet', 'Inception',
  'YOLO', 'Transformer', 'Encoder', 'Decoder', 'Attention',
  'Dataset', 'DataLoader', 'Optimizer', 'Scheduler',
  'Classifier', 'Regressor', 'Network', 'Model', 'Layer',
  // 机器学习缩写词（保持完整）
  'SVM', 'PCA', 'KMeans', 'GMM', 'LDA', 'NLP', 'CNN', 'RNN', 'LSTM', 'GAN'
];

// 类名到文件名的转换 (PascalCase → snake_case)
// 保留常见术语不拆分，例如 ImageNet → imagenet（而非 image_net）
function classNameToFileName(className) {
  // 术语到占位符的映射（使用不会被拆分的格式）
  const termMap = {};
  for (let i = 0; i < KNOWN_TERMS.length; i++) {
    termMap[KNOWN_TERMS[i]] = `__TERM${i}__`;
  }

  let result = className;

  // 先用占位符保护已知术语（大写匹配，保护整个单词）
  for (const [term, placeholder] of Object.entries(termMap)) {
    result = result.replace(term, placeholder);
  }

  // 在占位符前后添加下划线边界（确保术语与其他部分分隔）
  result = result
    .replace(/([a-z])(__TERM)/g, '$1_$2')  // 小写后接占位符时加下划线
    .replace(/(__\w+__)(__TERM)/g, '$1_$2')  // 占位符后接占位符时加下划线
    .replace(/(__TERM\d+__)([A-Z])/g, '$1_$2');  // 占位符后接大写时加下划线

  // 对剩余部分进行 snake_case 转换
  result = result
    .replace(/([a-z])([A-Z])/g, '$1_$2')  // 小写后接大写时加下划线
    .toLowerCase();

  // 恢复已知术语（占位符 → 小写术语）
  for (let i = 0; i < KNOWN_TERMS.length; i++) {
    const term = KNOWN_TERMS[i].toLowerCase();
    const placeholder = `__term${i}__`;  // 占位符也被 toLowerCase 了
    result = result.replace(placeholder, term);
  }

  // 清理多余的下划线
  result = result.replace(/_+/g, '_').replace(/^_|_$/g, '');

  return result;
}

// 改进的类/函数定义提取（处理嵌套结构）
// 支持提取 class 定义和顶层 def 函数定义
function extractClassDefinition(code, name) {
  const lines = code.split('\n');
  const result = [];
  let foundDef = false;
  let defBaseIndent = -1;
  let isClass = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();
    const lineIndent = line.search(/\S/);

    // 跳过开头的 import 语句（不属于定义）
    if (!foundDef && (/^(import|from)\s+/.test(trimmedLine) || trimmedLine === '')) {
      continue;
    }

    // 查找目标定义（class 或 def）
    if (!foundDef) {
      const classMatch = trimmedLine.match(/^class\s+(\w+)/);
      const funcMatch = trimmedLine.match(/^def\s+(\w+)/);
      if ((classMatch && classMatch[1] === name) || (funcMatch && funcMatch[1] === name)) {
        foundDef = true;
        isClass = !!classMatch;
        defBaseIndent = lineIndent;
        result.push(line);
        continue;
      }
    } else {
      // 已经找到定义，收集内容

      // 空行直接添加
      if (trimmedLine === '') {
        result.push(line);
        continue;
      }

      // 检查是否遇到同级的顶级定义（定义结束）
      if (lineIndent === defBaseIndent) {
        // 遇到新的顶级 class/def 或顶层代码，定义结束
        if (/^class\s+\w/.test(trimmedLine) || /^def\s+\w/.test(trimmedLine)) {
          break;
        }
        break;
      }

      // 检查是否是顶层代码（比定义缩进更少）
      if (lineIndent < defBaseIndent) {
        break;
      }

      result.push(line);
    }
  }

  // 移除末尾多余的空行
  while (result.length > 0 && result[result.length - 1].trim() === '') {
    result.pop();
  }

  return result.join('\n');
}

// 处理单个文件
function processFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const results = [];

  // 正则匹配 extract-class="ClassName" 或 extract-class="ClassA, ClassB"
  // 支持格式: ```python runnable [gpu] [timeout=xxx] extract-class="ClassName"
  // 多类格式: ```python runnable [gpu] [timeout=xxx] extract-class="ClassA, ClassB"
  const codeBlockRegex = /```python\s+runnable[\s\w=]*extract-class="([^"]+)"[\s\w=]*\n([\s\S]*?)```/g;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    const classListStr = match[1];
    const code = match[2];

    // 支持逗号分隔的多类名（向前兼容：单类名也走同一逻辑）
    const classNames = classListStr.split(',').map(s => s.trim()).filter(s => s.length > 0);

    // 收集同一代码块中所有提取到的定义
    const definitions = [];
    for (const name of classNames) {
      const definition = extractClassDefinition(code, name);
      if (definition) {
        definitions.push({ name, code: definition });
        console.log(`  ✓ 找到定义: ${name}`);
      } else {
        console.log(`  ⚠ 未能提取定义: ${name}`);
      }
    }

    if (definitions.length > 0) {
      // 多类提取时，所有定义合并为一个结果条目
      // 第一个名称作为文件名依据
      results.push({
        className: definitions[0].name,
        definitions: definitions,
        sourceCode: code  // 保留原始代码块，用于提取 import 语句
      });
    }
  }

  return results;
}

// 从原始代码块中提取 import 语句
function extractImportsFromSource(sourceCode) {
  const imports = [];
  for (const line of sourceCode.split('\n')) {
    const trimmed = line.trim();
    if (/^(import|from)\s+/.test(trimmed)) {
      imports.push(trimmed);
    }
  }
  return imports;
}

// 根据代码内容自动检测需要的导入
function detectAutoImports(combinedCode) {
  const imports = [];

  // PyTorch 相关
  if (combinedCode.includes('torch.') || combinedCode.includes('nn.') || combinedCode.includes('Tensor')) {
    imports.push('import torch');
  }
  if (combinedCode.includes('nn.') || combinedCode.includes('nn.Module') || combinedCode.includes('Sequential') ||
      combinedCode.includes('Conv2d') || combinedCode.includes('Linear') || combinedCode.includes('ReLU') ||
      combinedCode.includes('Dropout') || combinedCode.includes('MaxPool2d') || combinedCode.includes('AdaptiveAvgPool2d')) {
    imports.push('import torch.nn as nn');
  }
  if (combinedCode.includes('Dataset') || combinedCode.includes('DataLoader')) {
    imports.push('from torch.utils.data import Dataset, DataLoader');
  }
  if (combinedCode.includes('F.') || combinedCode.includes('functional')) {
    imports.push('import torch.nn.functional as F');
  }

  // 图像处理
  if (combinedCode.includes('Image')) {
    imports.push('from PIL import Image');
  }
  if (combinedCode.includes('transforms.')) {
    imports.push('from torchvision import transforms');
  }

  // 数值计算
  if (combinedCode.includes('np.') || combinedCode.includes('numpy')) {
    imports.push('import numpy as np');
  }

  // 文件系统
  if (combinedCode.includes('os.') || combinedCode.includes('os.path')) {
    imports.push('import os');
  }

  // JSON 处理
  if (combinedCode.includes('json.') || /\bjson\b/.test(combinedCode)) {
    imports.push('import json');
  }

  // 正则表达式
  if (combinedCode.includes('re.') || /\bre\b/.test(combinedCode)) {
    imports.push('import re');
  }

  // collections 模块
  if (combinedCode.includes('Counter') || combinedCode.includes('defaultdict') || combinedCode.includes('deque')) {
    imports.push('from collections import Counter, defaultdict, deque');
  }

  // math 模块
  if (combinedCode.includes('math.') || /\bmath\b/.test(combinedCode)) {
    imports.push('import math');
  }

  // dataclasses
  if (combinedCode.includes('@dataclass') || combinedCode.includes('dataclass')) {
    imports.push('from dataclasses import dataclass');
  }

  // typing
  if (combinedCode.includes('Optional') || combinedCode.includes('Tuple') || combinedCode.includes('List') || combinedCode.includes('Dict')) {
    imports.push('from typing import Optional, Tuple, List, Dict');
  }

  return imports;
}

// 生成 Python 模块文件
// definitions: [{name, code}, ...] — 提取到的类/函数定义
// sourceCode: 原始代码块（用于提取 import 语句）
function generateModuleFile(definitions, sourceCode, moduleDir) {
  const firstName = definitions[0].name;
  const fileName = classNameToFileName(firstName) + '.py';
  const modulePath = path.join(SHARED_MODULES_DIR, moduleDir, fileName);

  // 合并所有定义的代码
  const combinedCode = definitions.map(d => d.code).join('\n\n\n');

  // 从原始代码块提取显式 import
  const explicitImports = extractImportsFromSource(sourceCode);

  // 自动检测需要的导入
  const autoImports = detectAutoImports(combinedCode);

  // 合并并去重（显式 import 优先，补充自动检测到的）
  const importSet = new Set(explicitImports);
  for (const imp of autoImports) {
    // 仅添加自动检测到的、与已有 import 不冲突的
    const impModule = imp.replace(/^from\s+(\S+)\s+.*/, '$1').replace(/^import\s+(\S+).*/, '$1');
    let conflict = false;
    for (const existing of importSet) {
      const existingModule = existing.replace(/^from\s+(\S+)\s+.*/, '$1').replace(/^import\s+(\S+).*/, '$1');
      if (impModule === existingModule) {
        conflict = true;
        break;
      }
    }
    if (!conflict) {
      importSet.add(imp);
    }
  }

  // 排序：import 在前，from 在后
  const sortedImports = [...importSet].sort((a, b) => {
    if (a.startsWith('from') && !b.startsWith('from')) return 1;
    if (!a.startsWith('from') && b.startsWith('from')) return -1;
    return a.localeCompare(b);
  });

  const importsSection = sortedImports.length > 0
    ? sortedImports.join('\n') + '\n\n'
    : '';

  const names = definitions.map(d => d.name).join(', ');
  const content = `# ${names} 定义
# 从文档自动提取生成

${importsSection}${combinedCode}
`;

  return { path: modulePath, content };
}

// 可选依赖列表（这些包可能未安装）
const OPTIONAL_DEPENDENCIES = ['lmdb', 'redis', 'pymongo', 'mysql', 'psycopg2'];

// 更新 __init__.py（完全重新生成，扫描目录中所有 .py 文件）
function updateInitPy(moduleDir, classNames) {
  const initPath = path.join(SHARED_MODULES_DIR, moduleDir, '__init__.py');
  const moduleFullPath = path.join(SHARED_MODULES_DIR, moduleDir);

  if (!fs.existsSync(initPath)) {
    return;
  }

  // 扫描目录下所有 .py 文件（排除 __init__.py 和 __pycache__）
  const existingFiles = fs.readdirSync(moduleFullPath)
    .filter(f => f.endsWith('.py') && f !== '__init__.py');

  // 从 .py 文件内容中提取类名和函数名，并映射到实际文件名
  const fileClassMap = {};  // 文件名 -> 名称列表
  const fileDepsMap = {};   // 文件名 -> 可选依赖列表
  const allClassNames = [];

  for (const file of existingFiles) {
    const filePath = path.join(moduleFullPath, file);
    const content = fs.readFileSync(filePath, 'utf-8');
    const classMatches = content.match(/^class\s+(\w+)/gm);
    const funcMatches = content.match(/^def\s+(\w+)/gm);
    const classes = classMatches ? classMatches.map(m => m.replace('class ', '')) : [];
    const funcs = funcMatches ? funcMatches.map(m => m.replace('def ', '')) : [];
    const names = [...classes, ...funcs];

    if (names.length > 0) {
      fileClassMap[file] = names;
      allClassNames.push(...names);

      // 检测文件中的可选依赖
      // 检测方式：import xxx 或 from xxx 或 xxx.（模块使用）
      const deps = [];
      for (const dep of OPTIONAL_DEPENDENCIES) {
        if (content.includes(`import ${dep}`) ||
            content.includes(`from ${dep}`) ||
            content.includes(`${dep}.`)) {
          deps.push(dep);
        }
      }
      if (deps.length > 0) {
        fileDepsMap[file] = deps;
      }
    }
  }

  // 合并新提取的类名
  allClassNames.push(...classNames);
  const uniqueClassNames = [...new Set(allClassNames)];

  // 生成导入语句（按文件分组，对有可选依赖的使用 try-except）
  const importGroups = {};  // 文件名 -> { classes, hasOptionalDep }

  for (const className of uniqueClassNames) {
    // 查找类所在的实际文件
    let actualFileName = null;
    for (const [file, classes] of Object.entries(fileClassMap)) {
      if (classes.includes(className)) {
        actualFileName = file.replace('.py', '');
        break;
      }
    }

    // 如果没找到，使用推断的文件名
    if (!actualFileName) {
      actualFileName = classNameToFileName(className);
    }

    // 初始化文件组
    if (!importGroups[actualFileName]) {
      const hasOptionalDep = fileDepsMap[`${actualFileName}.py`];
      importGroups[actualFileName] = { classes: [], hasOptionalDep };
    }
    importGroups[actualFileName].classes.push(className);
  }

  // 生成导入语句
  const importLines = [];
  for (const [fileName, { classes, hasOptionalDep }] of Object.entries(importGroups)) {
    const classList = classes.join(', ');
    const importStmt = `from .${fileName} import ${classList}`;

    if (hasOptionalDep) {
      // 使用 try-except 包裹有可选依赖的导入
      importLines.push(`try:`);
      importLines.push(`    ${importStmt}`);
      importLines.push(`except ImportError:`);
      importLines.push(`    pass  # 可选依赖 ${hasOptionalDep.join(', ')} 未安装`);
    } else {
      importLines.push(importStmt);
    }
  }

  // 生成 __all__ 列表
  const allList = uniqueClassNames.map(n => `'${n}'`).join(', ');

  const content = `# ${moduleDir.toUpperCase()} 模块
${importLines.join('\n')}

__all__ = [${allList}]
`;

  fs.writeFileSync(initPath, content);
}

// 主函数
function main() {
  console.log('开始提取共享模块...\n');

  // 确保共享模块目录存在
  if (!fs.existsSync(SHARED_MODULES_DIR)) {
    console.log('错误: shared 目录不存在');
    process.exit(1);
  }

  // 扫描所有文档目录（支持自动推断）
  const moduleClasses = {}; // 按模块分组收集类名

  // 遍历 docs 下的所有子目录
  const categories = fs.readdirSync(DOCS_DIR).filter(f => {
    const fullPath = path.join(DOCS_DIR, f);
    return fs.statSync(fullPath).isDirectory() &&
           !f.startsWith('.') &&
           f !== 'superpowers';  // 排除 superpowers（内部文档）
  });

  for (const category of categories) {
    const categoryPath = path.join(DOCS_DIR, category);
    const chapters = fs.readdirSync(categoryPath).filter(f => {
      const fullPath = path.join(categoryPath, f);
      return fs.statSync(fullPath).isDirectory() && !f.startsWith('.');
    });

    for (const chapter of chapters) {
      const docPath = `${category}/${chapter}`;
      const chapterPath = path.join(DOCS_DIR, docPath);

      console.log(`\n扫描 ${docPath}/`);

      const files = fs.readdirSync(chapterPath).filter(f => f.endsWith('.md'));

      for (const file of files) {
        const filePath = path.join(chapterPath, file);
        const classes = processFile(filePath);

        if (classes.length > 0) {
          // 使用推断函数获取模块路径（支持显式映射 + 自动推断）
          const moduleDir = inferModulePath(docPath);
          const mappingType = CHAPTER_MAPPING[docPath] ? '(显式映射)' : '(自动推断)';
          console.log(`  模块路径: shared/${moduleDir}/ ${mappingType}`);

          // 确保模块目录和 __init__.py 存在
          const fullModuleDir = path.join(SHARED_MODULES_DIR, moduleDir);
          if (!fs.existsSync(fullModuleDir)) {
            fs.mkdirSync(fullModuleDir, { recursive: true });
            console.log(`    创建目录: ${moduleDir}/`);
          }
          const initPyPath = path.join(fullModuleDir, '__init__.py');
          if (!fs.existsSync(initPyPath)) {
            fs.writeFileSync(initPyPath, '__all__ = []\n');
          }

          if (!moduleClasses[moduleDir]) {
            moduleClasses[moduleDir] = [];
          }

          for (const { className, definitions, sourceCode } of classes) {
            // 生成模块文件
            const { path: modulePath, content } = generateModuleFile(definitions, sourceCode, moduleDir);
            fs.writeFileSync(modulePath, content);
            console.log(`    写入: ${path.relative(PROJECT_ROOT, modulePath)}`);

            // 收集所有定义名称（类名 + 函数名），用于更新 __init__.py
            for (const def of definitions) {
              moduleClasses[moduleDir].push(def.name);
            }
          }
        }
      }
    }
  }

  // 更新各模块的 __init__.py
  console.log('\n更新 __init__.py 文件...');
  for (const [moduleDir, classNames] of Object.entries(moduleClasses)) {
    updateInitPy(moduleDir, [...new Set(classNames)]);
    console.log(`  ✓ ${moduleDir}/__init__.py`);
  }

  // 更新顶层 __init__.py（完全重新生成）
  const topInitPath = path.join(SHARED_MODULES_DIR, '__init__.py');
  if (fs.existsSync(topInitPath)) {
    // 获取所有子模块目录（排除 __pycache__）
    const subModules = fs.readdirSync(SHARED_MODULES_DIR).filter(f => {
      const fullPath = path.join(SHARED_MODULES_DIR, f);
      return fs.statSync(fullPath).isDirectory() &&
             !f.startsWith('.') &&
             f !== '__pycache__';
    });

    // 生成简洁的 __init__.py：不使用 from .xxx import *（避免可选依赖问题）
    // 用户需要显式导入，如 from shared.sequence_models import PoetryLSTM
    const content = `# shared 模块包初始化
# 包含统计学习系列文档中可复用的类定义
#
# 使用方式：显式导入需要的模块
#   from shared.sequence_models import PoetryLSTM, PoetryDataset
#   from shared.cnn import AlexNet
#   from shared.linear import LogisticRegression

__all__ = [${subModules.map(m => `'${m}'`).join(', ')}]
`;

    fs.writeFileSync(topInitPath, content);
    console.log('  ✓ shared/__init__.py');
  }

  console.log('\n✓ 提取完成！');
}

main();