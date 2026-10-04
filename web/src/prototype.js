// PROTOTYPE: switches the layout variant with ?variant=A|B|C|D|E, arrow keys or the bar. Throwaway.
const NAMES = { A: "Split (current)", B: "Band", C: "Diptych", D: "Rail", E: "Compact readout" };
const KEYS = Object.keys(NAMES);
const url = new URL(location.href);
let current = KEYS.includes(url.searchParams.get("variant")) ? url.searchParams.get("variant") : "A";

const bar = document.createElement("div");
bar.className = "proto-bar";
const prev = document.createElement("button");
prev.type = "button";
prev.textContent = "←";
const next = document.createElement("button");
next.type = "button";
next.textContent = "→";
const label = document.createElement("output");
bar.append(prev, label, next);
document.body.append(bar);

function apply(key) {
  current = key;
  document.documentElement.dataset.variant = key;
  url.searchParams.set("variant", key);
  history.replaceState(null, "", url);
  label.textContent = `${key} (${NAMES[key]})`;
}
const step = (d) => apply(KEYS[(KEYS.indexOf(current) + d + KEYS.length) % KEYS.length]);
prev.addEventListener("click", () => step(-1));
next.addEventListener("click", () => step(1));
addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, textarea, [contenteditable]")) return;
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(1);
});
apply(current);
