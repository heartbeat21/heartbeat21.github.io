// docs/.vitepress/config.js
import mathjax3 from 'markdown-it-mathjax3';
import { flexibleMath } from './plugins/flexible-math.js';
import { mpeSyntax } from './plugins/mpe-syntax.js';

const customElements = [
  'math',
  'maction',
  'maligngroup',
  'malignmark',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mi',
  'mlongdiv',
  'mmultiscripts',
  'mn',
  'mo',
  'mover',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'ms',
  'mscarries',
  'mscarry',
  'mscarries',
  'msgroup',
  'mstack',
  'mlongdiv',
  'msline',
  'mstack',
  'mspace',
  'msqrt',
  'msrow',
  'mstack',
  'mstack',
  'mstyle',
  'msub',
  'msup',
  'msubsup',
  'mtable',
  'mtd',
  'mtext',
  'mtr',
  'munder',
  'munderover',
  'semantics',
  'math',
  'mi',
  'mn',
  'mo',
  'ms',
  'mspace',
  'mtext',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'msqrt',
  'mstyle',
  'mmultiscripts',
  'mover',
  'mprescripts',
  'msub',
  'msubsup',
  'msup',
  'munder',
  'munderover',
  'none',
  'maligngroup',
  'malignmark',
  'mtable',
  'mtd',
  'mtr',
  'mlongdiv',
  'mscarries',
  'mscarry',
  'msgroup',
  'msline',
  'msrow',
  'mstack',
  'maction',
  'semantics',
  'annotation',
  'annotation-xml',
  'mjx-container',
  'mjx-assistive-mml',
];

export default {
  // 站点级选项
  debug: true,
  title: "gujc的博客", // 网站标题
  description: "Some blobs of thought, code, and learning", // 网站描述
  markdown: {
      // math: true,  // 启用数学公式支持（使用 KaTeX）
      config: (md) => {
        md.use(mpeSyntax);
        md.use(flexibleMath);
        md.use(mathjax3, {
          packages: ['base'],
          // ✅ 显式启用 \(\) 和 \[\]
          tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [
              ['$$', '$$'],
              ['\\[', '\\]']
            ],
          },
          // 可选：启用更多 TeX 功能
          options: {
            processEscapes: true,
            processEnvironments: true
          }
        });
        // console.log('\n=== MarkdownIt Block Rulers ===')
        // md.block.ruler.getRules('').forEach((ruleName, index) => {
        //   console.log(`${index}: ${ruleName.name} ()`)
        // });
        // console.log('\n=== MarkdownIt Inline Rulers ===');
        // md.inline.ruler.getRules('').forEach((rule, index) => {
        //   // 尝试打印更多的信息，比如函数名（如果有的话）
        //   let ruleInfo = `${index}: ${rule.name || 'anonymous'} (${typeof rule})`;
        //   console.log(ruleInfo);
        // });
      },
    },
    vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag)
      }
    }
  },

  themeConfig: {
    nav: [
      {
        text: '文档目录',
        items: [
          { text: "mamba", link: "/mamba/" }
        ]
      }
    ],
    // sidebar: [
    //   {
    //     text: 'Examples',
    //     items: [
    //       { text: 'Markdown Examples', link: '/markdown-examples' },
    //       { text: 'Runtime API Examples', link: '/api-examples' }
    //     ]
    //   }
    // ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ],
    //本地搜索
    search: { 
      provider: 'local'
    }, 
  },
};
