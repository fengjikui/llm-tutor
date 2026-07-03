(() => {
  const THEME_KEY = "llmTutorCodeTheme";
  const THEMES = new Set(["dark", "light"]);
  const root = document.documentElement;

  function storedTheme() {
    try {
      const value = window.localStorage.getItem(THEME_KEY);
      return THEMES.has(value) ? value : "dark";
    } catch {
      return "dark";
    }
  }

  function currentTheme() {
    return root.dataset.codeTheme === "light" ? "light" : "dark";
  }

  function applyTheme(theme) {
    const next = THEMES.has(theme) ? theme : "dark";
    root.dataset.codeTheme = next;
    try {
      window.localStorage.setItem(THEME_KEY, next);
    } catch {
      // Storage can be unavailable in private or restricted browser contexts.
    }
    document.querySelectorAll(".code-theme-button").forEach((button) => {
      const toTheme = next === "dark" ? "浅色" : "深色";
      button.setAttribute("aria-label", `切换为${toTheme}代码主题`);
      button.setAttribute("title", `切换为${toTheme}代码主题`);
    });
  }

  function nextTheme() {
    return currentTheme() === "dark" ? "light" : "dark";
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
    return Promise.resolve();
  }

  function markCopied(button) {
    button.classList.add("is-copied");
    button.setAttribute("aria-label", "已复制代码");
    button.setAttribute("title", "已复制代码");
    window.setTimeout(() => {
      button.classList.remove("is-copied");
      button.setAttribute("aria-label", "复制代码");
      button.setAttribute("title", "复制代码");
    }, 1400);
  }

  function makeButton(className, label) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `code-tool-button ${className}`;
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
    return button;
  }

  function hostFor(pre) {
    const figure = pre.closest("figure.code-block");
    if (figure) {
      figure.classList.add("code-tool-host");
      return figure;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "code-pre-wrap";
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);
    return wrapper;
  }

  function hasToolbar(host) {
    return Array.from(host.children).some((child) => child.classList.contains("code-toolbar"));
  }

  function enhance(pre) {
    if (pre.dataset.codeEnhanced === "true") {
      return;
    }
    pre.dataset.codeEnhanced = "true";

    const host = hostFor(pre);
    if (hasToolbar(host)) {
      return;
    }

    const toolbar = document.createElement("div");
    toolbar.className = "code-toolbar";

    const themeButton = makeButton("code-theme-button", "切换代码主题");
    themeButton.addEventListener("click", () => applyTheme(nextTheme()));

    const copyButton = makeButton("code-copy-button", "复制代码");
    copyButton.addEventListener("click", async () => {
      const code = pre.querySelector("code");
      const text = code ? code.innerText : pre.innerText;
      await copyText(text.replace(/\n$/, ""));
      markCopied(copyButton);
    });

    toolbar.append(themeButton, copyButton);
    host.appendChild(toolbar);
  }

  function init() {
    document.querySelectorAll("pre").forEach(enhance);
    applyTheme(storedTheme());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
