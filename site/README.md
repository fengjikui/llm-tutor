# 专栏静态站点

这是 `llm_tutor` 的个人网站专栏第一版，使用纯静态 HTML/CSS/JS，便于嵌入现有个人站点或直接通过 GitHub Pages 发布。

本地预览可以直接打开 `site/index.html`。如果浏览器安全策略影响本地资源加载，也可以在仓库根目录运行：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000/site/`。

## 全文渲染路线

当前首页是高审美的专栏入口，章节正文仍跳转到 GitHub Markdown。等教程内容稳定后，网站端会升级成完整阅读体验：

- 直接渲染 `tutorials/*.md` 正文，而不只是章节卡片。
- 保留 GitHub 链接，方便读者查看源码、提交 issue 或直接读仓库版。
- 为每篇文章生成目录、上一篇/下一篇、代码块高亮、参考资料区和实验命令区。
- 支持公式、Mermaid 图、可折叠解释、学习者疑问卡片和实验产物截图。
- 首页继续保持现在的审美密度，正文页则采用更适合长文阅读的版式。

实现时优先采用静态生成：读取 frontmatter 和 Markdown，构建 `/site/chapters/<slug>.html`，并在 GitHub Pages 或其他静态托管平台直接发布。
