// Configuration - update this to your backend URL
const BACKEND_BASE = "https://informed-treasures-coalition-connected.trycloudflare.com";

const form = document.getElementById("form");
const msg = document.getElementById("msg");

function formatScore(score) {
    return (score * 100).toFixed(2) + "%";
}

function formatRank(rank) {
    return rank.toFixed(2);
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.className = "muted";
    msg.textContent = "Submitting…";

    const name = document.getElementById("name").value.trim();
    const file = document.getElementById("file").files[0];
    if (!name || !file) {
        msg.className = "bad";
        msg.textContent = "Please provide both name and file";
        return;
    }

    const fd = new FormData();
    fd.append("name", name);
    fd.append("file", file);

    try {
        const r = await fetch(`${BACKEND_BASE}/api/submit`, {
            method: "POST",
            body: fd,
        });

        const data = await r.json().catch(() => ({}));

        if (!r.ok) {
            throw new Error(data.detail || `HTTP ${r.status}`);
        }

        // Display all scores
        msg.className = "ok";
        msg.innerHTML = `
      <strong>Submission successful!</strong><br>
      <div style="margin-top: 0.5rem;">
        Amplitude Shift: ${formatScore(data.score_dataset1)}<br>
        Frequency Shift: ${formatScore(data.score_dataset2)}<br>
        Noise Shift: ${formatScore(data.score_dataset3)}<br>
        Trend Shift: ${formatScore(data.score_dataset4)}<br>
        <strong>Average Rank: ${formatRank(data.avg_rank || 0)}</strong>
      </div>
      <div style="margin-top: 0.5rem; font-size: 0.9em; color: #b5b5b5;">
        Submission ID: ${data.submission_id}<br>
        <a href="../">View leaderboard →</a>
      </div>
    `;
    } catch (err) {
        msg.className = "bad";
        msg.textContent = `Error: ${err.message}`;
    }
});
