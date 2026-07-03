const chapters = window.LLM_TUTOR_CHAPTERS || [];
const list = document.querySelector("#chapter-list");
const tabs = document.querySelectorAll(".phase-tab");

function renderChapters(phase = "all") {
  const selected = phase === "all" ? chapters : chapters.filter((chapter) => chapter.phase === phase);
  list.innerHTML = selected
    .map(
      (chapter) => `
        <a class="chapter-row" href="${chapter.url}">
          <span class="chapter-no">${chapter.no}</span>
          <span class="chapter-main">
            <h3>${chapter.title}</h3>
            <p>${chapter.summary}</p>
          </span>
          <span class="chapter-meta">
            ${chapter.tags.map((tag) => `<span>${tag}</span>`).join("")}
          </span>
        </a>
      `
    )
    .join("");
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("is-active"));
    tab.classList.add("is-active");
    renderChapters(tab.dataset.phase);
  });
});

if (list) {
  renderChapters();
}
