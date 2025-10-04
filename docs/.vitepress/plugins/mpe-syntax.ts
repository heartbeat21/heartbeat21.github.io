// docs/.vitepress/plugins/mpe-syntax.ts

import { stat } from 'fs';
import type { PluginWithOptions } from 'markdown-it';

export const mpeSyntax: PluginWithOptions = (md) => {
  // 1. 注册预处理器，在解析前修改 Markdown 源码
  md.core.ruler.before('normalize', 'mpe-syntax', (state) => {
    let src = state.src;

    // ✅ 1. 处理 ----...----
    src = src.replace(/----([\w\W]+?)----/g, (match, p1) => {
      const lines = p1
        .replace(/^(\w+):\s+([^\n]+)(\n|$)/gm, (match, k, v) => {
          return `${k} : ${v}<br>`;
        });
      return `<div class='my-front'>${lines}</div>`;
    });

    // ✅ 2. 处理 %%date 和 %%time
    const now = new Date();
    const date = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

    src = src.replace(/%%date/g, date + ': ');
    src = src.replace(/%%time/g, time);

    // ✅ 3. 处理 \ref{...}
    src = src.replace(/\\ref\{([^}]*)\}/g, (match, p1) => {
      return `<a href="#${p1}">${p1}</a>`;
    });

    // // ✅ 4. 处理 \label{content}{id}
    // src = src.replace(/\\label\{([^}]*)\}\{([^}]*)\}/g, (match, content, id) => {
    //   return `<span id="${id}">${content}</span>`;
    // });

    // ✅ 5. 处理 <email@domain.com> → 邮箱链接
    src = src.replace(/<([^>@\s]+@[^>\s]+)>/g, (match, email) => {
      return `<a href="mailto:${email}" target="_blank" class="email-link">${email}</a>`;
    });

    // 更新源码
    state.src = src;
  });

  // md.core.ruler.push('debug-tokens', (state) => {
  //   // if (state.env.isDebug !== true) return;

  //   console.log('\n=== MarkdownIt Token Stream ===');
  //   state.tokens.forEach((token, index) => {
  //     const indent = '  '.repeat(token.level);
  //     console.log(`${index}: ${indent}${token.type} (${token.tag})`, {
  //       content: token.content?.substring(0, 50),
  //       level: token.level,
  //       nesting: token.nesting,
  //       children: token.children?.length,
  //       attrs: token.attrs,
  //     });
  //   });
  // });

  md.block.ruler.after('fence', 'mpe-label-block', function (state, startLine, endLine, silent) {
    const start = state.bMarks[startLine] + state.tShift[startLine];
    const max = state.eMarks[startLine];

    // 检查当前行是否以 \label{ 开头
    const slice = state.src.slice(start, start + 7);
    if (slice !== '\\label{') return false;

    if (silent) return false;

    let line = startLine;
    let pos = start + 7;
    let depth = 1;
    const maxPos = state.eMarks[endLine];

    // 从当前行开始扫描，直到 depth === 0
    const contentStart = pos;
    while (pos < maxPos && depth > 0) {
      const ch = state.src.charCodeAt(pos);

      // 跳过转义
      if (ch === 0x5c && pos + 1 < maxPos) {
        pos += 2;
        continue;
      }

      if (ch === 0x7b) depth++;   // {
      else if (ch === 0x7d) depth--; // }

      pos++;

      // 到行尾，换行
      if (pos >= state.eMarks[line]) {
        line++;
        if (line > endLine) break;
        pos = state.bMarks[line] + state.tShift[line];
      }
    }
    
    if (depth !== 0) return false; // 括号未闭合

    // 现在 pos 指向第一个 } 后的位置
    const contentEnd = pos - 2; // 回退到 }
    const content = state.src.slice(contentStart, contentEnd).trim().slice(2, -2);
    // 解析 {id}
    pos--;
    if (pos >= maxPos || state.src.slice(pos, pos + 2) !== '}{') return false;
    pos += 2; // 跳过 }{

    // 解析 {id}
    const idStart = pos;
    depth = 1;
    while (pos < maxPos && depth > 0) {
      const ch = state.src.charCodeAt(pos);
      if (ch === 0x5c && pos + 1 < maxPos) {
        pos += 2;
        continue;
      }
      if (ch === 0x7b) depth++;
      else if (ch === 0x7d) depth--;
      pos++;
    }
    if (depth !== 0) return false;

    // 触发 mathjex

    const id = state.src.slice(idStart, pos - 1).replace(/[^a-zA-Z0-9\-_:]/g, ' ');

    // 创建 token
    const tokenOpen = new state.Token('mpe_label_block_open', 'div', 1);
    tokenOpen.attrs = [['id', id]];
    tokenOpen.map = [startLine, line];
    tokenOpen.markup = '\\label';

    const tokenContent = new state.Token('math_block', 'math', 0);
    tokenContent.content = content;
    tokenContent.level = state.level + 1;
    tokenContent.block = true;

    tokenContent.children = [];
    tokenContent.level = state.level + 1;

    const tokenClose = new state.Token('mpe_label_block_close', 'div', -1);
    // console.log("mpe-syntax: ", tokenContent);

    // 插入 token
    state.tokens.push(tokenOpen);
    state.tokens.push(tokenContent);
    state.tokens.push(tokenClose);

    // 更新行号
    state.line = line + 1;

    return true;
  });

  // 渲染器
  md.renderer.rules.mpe_label_block_open = (tokens, idx) => {
    const token = tokens[idx];
    const id = token.attrGet('id');
    return `<div id="${id}">\n`;
  };

  md.renderer.rules.mpe_label_block_close = () => '</div>\n';
};