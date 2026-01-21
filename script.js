(function() {
    const input = document.getElementById("schedule-search");
    const table = document.querySelector("table.schedule");
    if (!input || !table) return;

    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll("tr"))
        .filter(tr => tr.id !== "schedule-empty"); // ignore empty-state row if present
    const emptyRow = document.getElementById("schedule-empty");
    const countEl = document.getElementById("schedule-count");

    function update() {
        const q = input.value.trim().toLowerCase();
        let shown = 0;

        for (const tr of rows) {
            const text = tr.textContent.toLowerCase();
            const match = q === "" || text.includes(q);
            tr.style.display = match ? "" : "none";
            if (match) shown++;
        }

        if (emptyRow) emptyRow.style.display = shown === 0 ? "" : "none";
        if (countEl) {
            countEl.textContent = q ? `${shown} / ${rows.length} shown` : "";
        }
    }

    input.addEventListener("input", update);
    update();
})();