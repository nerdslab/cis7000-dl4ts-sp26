// Configuration - update this to your backend URL
const BACKEND_BASE = "https://informed-treasures-coalition-connected.trycloudflare.com";
const WS_BASE = BACKEND_BASE.replace(/^https/, "wss").replace(/^http/, "ws");

// State
let entries = [];
let sortColumn = "avg_rank";
let sortDirection = "asc";

// WebSocket connection
let ws = null;
let reconnectTimeout = null;

function connectWebSocket() {
    const wsUrl = `${WS_BASE}/ws/leaderboard`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        updateStatus("Connected", "ok");
        if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
        }
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.entries) {
                entries = data.entries;
                renderTable();
            }
        } catch (err) {
            console.error("Error parsing WebSocket message:", err);
        }
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        updateStatus("Connection error", "bad");
    };

    ws.onclose = () => {
        updateStatus("Disconnected - reconnecting...", "bad");
        // Reconnect after 3 seconds
        reconnectTimeout = setTimeout(connectWebSocket, 3000);
    };
}

function updateStatus(text, className = "muted") {
    const status = document.getElementById("status");
    status.textContent = text;
    status.className = className;
}

function formatScore(score) {
    return (score * 100).toFixed(2) + "%";
}

function formatRank(rank) {
    return rank.toFixed(2);
}

function sortEntries() {
    entries.sort((a, b) => {
        let aVal = a[sortColumn];
        let bVal = b[sortColumn];

        // Handle numeric comparison
        if (typeof aVal === "number" && typeof bVal === "number") {
            return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
        }

        // Handle string comparison
        if (typeof aVal === "string" && typeof bVal === "string") {
            return sortDirection === "asc"
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal);
        }

        return 0;
    });
}

function handleSort(column) {
    if (sortColumn === column) {
        sortDirection = sortDirection === "asc" ? "desc" : "asc";
    } else {
        sortColumn = column;
        sortDirection = "asc";
    }
    renderTable();
}

function renderTable() {
    sortEntries();

    const tbody = document.getElementById("rows");
    const tbl = document.getElementById("tbl");

    if (entries.length === 0) {
        tbody.innerHTML = "<tr><td colspan='7' style='text-align: center; color: #b5b5b5;'>No submissions yet</td></tr>";
        tbl.style.display = "";
        return;
    }

    // Compute ranks based on avg_rank (ties get same rank, sequential numbering)
    const ranks = [];
    let currentRank = 1;
    let prevAvgRank = null;
    
    for (let i = 0; i < entries.length; i++) {
        const avgRank = entries[i].avg_rank;
        if (prevAvgRank !== null && avgRank !== prevAvgRank) {
            // Different avg_rank, so increment rank
            currentRank++;
        }
        ranks.push(currentRank);
        prevAvgRank = avgRank;
    }

    tbody.innerHTML = entries
        .map((e, idx) => {
            const rank = ranks[idx];
            const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : '';
            const rankDisplay = medal ? `${medal} ${rank}` : rank;
            
            return `
        <tr>
          <td>${rankDisplay}</td>
          <td>${escapeHtml(e.name)}</td>
          <td>${formatScore(e.score_dataset1)}</td>
          <td>${formatScore(e.score_dataset2)}</td>
          <td>${formatScore(e.score_dataset3)}</td>
          <td>${formatScore(e.score_dataset4)}</td>
          <td>${formatRank(e.avg_rank)}</td>
        </tr>
      `;
        })
        .join("");

    // Update header sort indicators
    updateSortIndicators();
    tbl.style.display = "";
}

function updateSortIndicators() {
    // Remove all sort indicators
    document.querySelectorAll("th").forEach((th) => {
        th.textContent = th.textContent.replace(/ [↑↓]/, "");
    });

    // Add sort indicator to current column
    const headers = {
        name: 1,
        score_dataset1: 2,
        score_dataset2: 3,
        score_dataset3: 4,
        score_dataset4: 5,
        avg_rank: 6,
    };

    const colIdx = headers[sortColumn];
    if (colIdx) {
        const th = document.querySelectorAll("th")[colIdx];
        if (th) {
            th.textContent += sortDirection === "asc" ? " ↑" : " ↓";
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Make headers clickable
function setupSortableHeaders() {
    const headers = document.querySelectorAll("th");
    const headerMap = {
        1: "name",
        2: "score_dataset1",
        3: "score_dataset2",
        4: "score_dataset3",
        5: "score_dataset4",
        6: "avg_rank",
    };

    headers.forEach((th, idx) => {
        const col = headerMap[idx];
        if (col) {
            th.style.cursor = "pointer";
            th.style.userSelect = "none";
            th.addEventListener("click", () => handleSort(col));
        }
    });
}

// Initialize
updateStatus("Connecting...", "muted");
setupSortableHeaders();
connectWebSocket();
