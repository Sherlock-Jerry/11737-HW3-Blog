// import { searchPlugin } from "@vuepress/plugin-search";
// import { defineUserConfig } from "vuepress";
// import theme from "./theme.js";

// export default defineUserConfig({
//   lang: "en-US",
//   title: "MLNLP Blog",
//   description: "A Blog for Machine Learning, Natural Language Processing, and Data Mining",
//   base: "/blog/",
//   theme,
//   plugins: [
//     searchPlugin({
//       // your options
//     }),
//   ],
// });

import { searchPlugin } from "@vuepress/plugin-search";
import { defineUserConfig } from "vuepress";
import theme from "./theme.js";

export default defineUserConfig({
  lang: "en-US",
  title: "mT5: A Massively Multilingual Pre-Trained Text-To-Text Transformer | 11737-HW3-Blog",
  description: "A Blog for Machine Learning, Natural Language Processing, and Data Mining",
  base: "/11737-HW3-Blog/",
  theme,
  head: [
    ['script', {
      async: true,
      src: 'https://polyfill.io/v3/polyfill.min.js?features=es6',
    }],
    ['script', {
      async: true,
      src: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js',
    }],
  ],
  plugins: [
    searchPlugin({
      // your options
    }),
  ],
});