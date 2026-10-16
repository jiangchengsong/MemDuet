"use strict";

(() => {
  const dialog = document.querySelector("#figure-dialog");
  const dialogImage = document.querySelector("#dialog-image");
  const dialogTitle = document.querySelector("#dialog-title");
  const closeButton = document.querySelector("#close-figure");
  let figureTrigger = null;

  document.querySelectorAll("[data-figure]").forEach((button) => {
    button.addEventListener("click", () => {
      figureTrigger = button;
      dialogImage.src = button.dataset.figure;
      dialogImage.alt = button.querySelector("img").alt;
      dialogTitle.textContent = button.dataset.title;
      dialog.showModal();
      document.body.classList.add("dialog-open");
      closeButton.focus();
    });
  });

  closeButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom) {
      dialog.close();
    }
  });
  dialog.addEventListener("close", () => {
    document.body.classList.remove("dialog-open");
    figureTrigger?.focus({ preventScroll: true });
  });

  const tableToggle = document.querySelector("#table-toggle");
  const tableRows = document.querySelectorAll("[data-extra-row]");
  const longContextNote = document.querySelector("#long-context-note");
  tableToggle.addEventListener("click", () => {
    const expanded = tableToggle.getAttribute("aria-expanded") !== "true";
    tableToggle.setAttribute("aria-expanded", String(expanded));
    tableRows.forEach((row) => { row.hidden = !expanded; });
    longContextNote.hidden = !expanded;
    tableToggle.firstChild.textContent = expanded
      ? "Show selected context lengths "
      : "Show all 10 context lengths ";
  });

  const copyButton = document.querySelector("#copy-bibtex");
  const feedback = document.querySelector("#copy-feedback");

  const copyWithSelection = (text) => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    let copied = false;
    try { copied = document.execCommand("copy"); }
    catch { copied = false; }
    textarea.remove();
    copyButton.focus({ preventScroll: true });
    return copied;
  };

  copyButton.addEventListener("click", async () => {
    const citation = document.querySelector("#citation-text").textContent.trim();
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(citation);
        copied = true;
      }
    } catch { copied = false; }
    if (!copied) copied = copyWithSelection(citation);
    copyButton.textContent = copied ? "Copied" : "Copy BibTeX";
    feedback.textContent = copied
      ? "BibTeX copied to clipboard."
      : "Copy is unavailable in this browser. Select the BibTeX above and copy it manually.";
  });
})();
