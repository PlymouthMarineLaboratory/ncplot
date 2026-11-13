(function () {
  "use strict";

  /* ---------- header scroll + mobile nav ---------- */
  var header = document.querySelector(".site-header");
  var navToggle = document.querySelector(".nav-toggle");
  var navLinks = document.querySelector(".nav-links");

  function onScroll() {
    if (!header) return;
    header.classList.toggle("is-scrolled", window.scrollY > 8);
  }
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      var open = navLinks.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    navLinks.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        navLinks.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  /* ---------- mark active nav link ---------- */
  var here = (location.pathname.split("/").pop() || "index.html");
  document.querySelectorAll(".nav-links a[data-page]").forEach(function (a) {
    if (a.getAttribute("data-page") === here) a.classList.add("active");
  });

  /* ---------- copy-to-clipboard for code cards ---------- */
  document.querySelectorAll("[data-copy-target]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var target = document.querySelector(btn.getAttribute("data-copy-target"));
      if (!target) return;
      var text = target.innerText;
      navigator.clipboard.writeText(text).then(function () {
        var label = btn.querySelector(".copy-label");
        var original = label ? label.textContent : null;
        if (label) label.textContent = "Copied!";
        setTimeout(function () {
          if (label && original !== null) label.textContent = original;
        }, 1600);
      });
    });
  });

  /* ---------- docs scrollspy ---------- */
  var tocLinks = document.querySelectorAll(".docs-toc a");
  if (tocLinks.length) {
    var targets = [];
    tocLinks.forEach(function (link) {
      var id = link.getAttribute("href").replace("#", "");
      var el = document.getElementById(id);
      if (el) targets.push({ link: link, el: el });
    });

    function updateToc() {
      var pos = window.scrollY + 120;
      var current = null;
      targets.forEach(function (t) {
        if (t.el.offsetTop <= pos) current = t;
      });
      tocLinks.forEach(function (l) { l.classList.remove("active"); });
      if (current) current.link.classList.add("active");
    }
    updateToc();
    window.addEventListener("scroll", updateToc, { passive: true });
  }

  /* ---------- docs version (latest PyPI release) ---------- */
  var versionEls = document.querySelectorAll("#ncplot-docs-version");
  if (versionEls.length) {
    fetch("https://pypi.org/pypi/ncplot/json")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        var version = data && data.info && data.info.version;
        if (!version) return;
        versionEls.forEach(function (el) {
          el.textContent = "v" + version;
          el.title = "NCPlot v" + version + " (latest PyPI release)";
        });
      })
      .catch(function () {
        /* PyPI unreachable (offline, blocked, rate-limited): leave the
           static fallback text in the HTML in place rather than showing
           an error or a stale version number. */
      });
  }

  /* ---------- back to top ---------- */
  var backToTop = document.querySelector(".back-to-top");
  if (backToTop) {
    window.addEventListener("scroll", function () {
      backToTop.classList.toggle("is-visible", window.scrollY > 600);
    }, { passive: true });
    backToTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
})();
