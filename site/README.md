# 专栏静态站点

这是 `llm_tutor` 的个人网站专栏第一版，使用纯静态 HTML/CSS/JS，便于嵌入现有个人站点或直接通过 GitHub Pages 发布。

章节正文来自仓库根目录的 `tutorials/*.md`，不要手写 `site/chapters/*.html`。修改 Markdown 后运行：

```bash
uv run python scripts/build_site.py
```

构建脚本会生成：

- `site/chapters/*.html`：可直接访问的章节正文页。
- `site/chapters-data.js`：首页章节列表数据。
- `site/chapters.json`：调试或后续搜索索引用的数据。

本地预览可以直接打开 `site/index.html`。如果浏览器安全策略影响本地资源加载，也可以在仓库根目录运行：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000/site/`。

## 全文渲染路线

当前首页是高审美的专栏入口，章节正文已经优先跳转到渲染后的网页。后续会继续增强完整阅读体验：

- 保留 GitHub 链接，方便读者查看源码、提交 issue 或直接读仓库版。
- 为每篇文章增强代码高亮、参考资料区和实验命令区。
- 支持公式、Mermaid 图、可折叠解释、学习者疑问卡片、全文搜索和实验产物截图。
- 首页继续保持现在的审美密度，正文页则采用更适合长文阅读的版式。

实现方式保持静态生成：读取 frontmatter 和 Markdown，构建 `/site/chapters/<slug>.html`，并在 GitHub Pages 或其他静态托管平台直接发布。
