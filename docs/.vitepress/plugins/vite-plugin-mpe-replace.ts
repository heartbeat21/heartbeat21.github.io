// docs/.vitepress/plugins/vite-plugin-mpe-replace.ts
import { readFileSync } from 'fs';
import type { Plugin } from 'vite';

export function mpeReplace(): Plugin {
  return {
    name: 'mpe-replace',

    // 使用 `transform` 钩子，在文件被读取后立即替换
    transform(code, id) {
      // 只处理 .md 文件
      if (!id.endsWith('.md') || id.includes('node_modules')) return code;

      let src = code;

      // 1. 处理 %%date / %%time
      const now = new Date();
      const date = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
      const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

      src = src.replace(/%%date/g, date);
      src = src.replace(/%%time/g, time);

      // 2. 处理 \ref{xxx}
      src = src.replace(/\\ref\{([^}]*)\}/g, '<a href="#$1">$1</a>');

      // 3. 处理 <email@domain.com>
      src = src.replace(/<([^>@\s]+@[^>\s]+)>/g, '<a href="mailto:$1" target="_blank" class="email-link">$1</a>');

      // 4. 处理 ----...----
      src = src.replace(/----([\w\W]+?)----/g, (match, p1) => {
        const lines = p1.replace(/^(\w+):\s+([^\n]+)(\n|$)/gm, (match, k, v) => {
          return `${k} : ${v}<br>`;
        });
        return `<div class='my-front'>${lines}</div>`;
      });

      return src;
    }
  };
}