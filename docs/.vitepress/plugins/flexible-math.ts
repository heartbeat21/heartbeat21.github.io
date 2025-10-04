// docs/.vitepress/plugins/flexible-math.ts

import type { PluginWithOptions } from 'markdown-it';
import type Token from 'markdown-it/lib/token.mjs';

export const flexibleMath: PluginWithOptions = (md) => {
  md.inline.ruler.before('text', 'math_paren', function (state, silent) {
    const pos = state.pos
    const src = state.src

    // 检查 \( 
    if (src.charCodeAt(pos) !== 0x5C /* \ */ || src.charCodeAt(pos + 1) !== 0x28 /* ( */) {
      return false
    }

    if (silent) return true

    const contentStart = pos + 2
    let level = 0
    let end = contentStart

    while (end < state.posMax) {
      const ch = src.charCodeAt(end)

      // 处理嵌套 {}
      if (ch === 0x7B) level++
      else if (ch === 0x7D) level--

      // 检查 \)
      else if (ch === 0x5C && end + 1 < state.posMax && src.charCodeAt(end + 1) === 0x29 /* ) */) {
        if (level <= 0) break
      }

      end++
    }

    // 检查是否以 \) 结尾
    if (src.slice(end, end + 2) !== '\\)') return false

    const token = state.push('math_inline', 'math', 0)
    token.content = src.slice(contentStart, end)
    token.markup = '\\('

    state.pos = end + 2
    return true
  })

  // === 2. 注册 \[\] 解析规则 ===
  md.block.ruler.before('paragraph', 'math_bracket', function (state, start, end, silent) {
    let next, lastPos;
    let found = false, pos = state.bMarks[start] + state.tShift[start], max = state.eMarks[start], lastLine = "";
    if (pos + 2 > max) {
        return false;
    }
    if (state.src.slice(pos, pos + 2) !== "\\[") {
        return false;
    }
    pos += 2;
    let firstLine = state.src.slice(pos, max);
    if (silent) {
        return true;
    }
    if (firstLine.trim().slice(-2) === "\\]") {
        // Single line expression
        firstLine = firstLine.trim().slice(0, -2);
        found = true;
    }
    for (next = start; !found;) {
        next++;
        if (next >= end) {
            break;
        }
        pos = state.bMarks[next] + state.tShift[next];
        max = state.eMarks[next];
        if (pos < max && state.tShift[next] < state.blkIndent) {
            // non-empty line with negative indent should stop the list:
            break;
        }
        if (state.src.slice(pos, max).trim().slice(-2) === "\\]") {
            lastPos = state.src.slice(0, max).lastIndexOf("\\]");
            lastLine = state.src.slice(pos, lastPos);
            found = true;
        }
    }
    state.line = next + 1;
    const token = state.push("math_block", "math", 0);
    token.block = true;
    token.content =
        (firstLine && firstLine.trim() ? firstLine + "\n" : "") +
            state.getLines(start + 1, next, state.tShift[start], true) +
            (lastLine && lastLine.trim() ? lastLine : "");
    console.log("block content: ", token.content);
    token.map = [start, state.line];
    token.markup = "\\[";
    return true;
  });

  md.inline.ruler.before(
    'text', // 在代码块之前处理
    'flexible-math',
    (state, silent) => {
      const { src, pos } = state;
      // 必须是 $ 开头
      if (src[pos] !== '$') return false;
      // 检查前一个字符是否为边界（防止匹配变量名如 $var）
      if (pos > 0) {
        const prev = src[pos - 1];
        if (/\w/.test(prev)) return false; // 前面是字母/数字，不匹配
      }
      // 查找下一个 $
      let nextPos = pos + 1;
      while (nextPos < src.length) {
        if (src[nextPos] === '\\' && nextPos + 1 < src.length) {
          nextPos += 2; // 跳过转义 \$ 
          continue;
        }
        if (src[nextPos] === '$') break;
        nextPos++;
      }

      if (nextPos >= src.length) {
        return false; // 未找到闭合 $
      }

      // 提取内容（包含空格）
      const content = src.slice(pos + 1, nextPos);

      // 关键：如果内容为空或只有空格，跳过
      if (!content.trim()) return false;

      // 非静默模式才添加 token
      if (!silent) {
        const token = state.push('math_inline', '', 0);
        // 这里 trim 掉首尾空格，只保留中间公式
        token.content = content.trim();
        token.markup = '$';
        token.info = 'tex';
        state.pos = nextPos + 1;
      }

      return true;
    }
  );

  // ✅ 确保 math_inline 被正确渲染（使用 KaTeX 行内模式）
  md.renderer.rules.math_inline = (tokens, idx) => {
    const content = tokens[idx].content;
    // 直接输出 \(\) 包裹的公式，由 KaTeX 渲染
    return `\\(${content}\\)`;
  };
};