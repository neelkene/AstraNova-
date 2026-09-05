/**
 * Frontend Application Logic & API Client
 * File: web/static/js/app.js
 * Project: AI-Driven Anomaly Detection in Component Burn-In & Screening (SIH 2026)
 */

document.addEventListener('DOMContentLoaded', () => {
    startLiveClock();
    initNavigation();
    initLiveTester();
    loadModelPerformance();
});

// State
let selectedComponentId = "SYN_C01216"; // Default healthy sample
let currentComponentData = null;
let allTestComponents = [];

/* ==============================================================================
   1. Navigation Tabs
   ============================================================================== */
function initNavigation() {
    const tabBtns = document.querySelectorAll('.nav-tab-btn');
    const viewPanels = document.querySelectorAll('.view-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetView = btn.getAttribute('data-view');
            
            tabBtns.forEach(b => b.classList.remove('active'));
            viewPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const activePanel = document.getElementById(targetView);
            if (activePanel) {
                activePanel.classList.add('active');
            }
        });
    });
}

/* ==============================================================================
   2. Live Component Tester & API Client
   ============================================================================== */
async function initLiveTester() {
    await loadTestComponentsList('all');
    setupFilterButtons();
    setupPresetButtons();
    setupScreeningActionButtons();
    
    // Auto-select initial preset
    selectComponent(selectedComponentId);
}

async function loadTestComponentsList(category = 'all') {
    const listEl = document.getElementById('component-scroll-list');
    listEl.innerHTML = '<div style="padding: 1rem; color: var(--text-muted); font-size: 0.8rem;">Loading genuine test set...</div>';

    try {
        const res = await fetch(`/api/test-components?category=${category}&limit=80`);
        const data = await res.json();
        allTestComponents = data.components || [];

        renderComponentList(allTestComponents);
    } catch (err) {
        listEl.innerHTML = '<div style="padding: 1rem; color: var(--status-reject); font-size: 0.8rem;">Failed to load test components</div>';
        console.error("API error loading test components:", err);
    }
}

function renderComponentList(components) {
    const listEl = document.getElementById('component-scroll-list');
    listEl.innerHTML = '';

    if (components.length === 0) {
        listEl.innerHTML = '<div style="padding: 1rem; color: var(--text-muted); font-size: 0.8rem;">No components found</div>';
        return;
    }

    components.forEach(c => {
        const item = document.createElement('div');
        item.className = `comp-list-item ${c.component_id === selectedComponentId ? 'selected' : ''}`;
        item.setAttribute('data-id', c.component_id);

        let badgeClass = 'badge-pass';
        if (c.category_key === 'drifting') badgeClass = 'badge-review';
        if (c.category_key === 'anomalous') badgeClass = 'badge-reject';

        item.innerHTML = `
            <div>
                <strong style="font-family: 'JetBrains Mono', monospace;">${c.component_id}</strong>
                <div style="font-size: 0.72rem; color: var(--text-muted);">${c.category.split(' ')[0]}</div>
            </div>
            <span class="badge ${badgeClass}" style="font-size: 0.68rem;">${c.iddq_drift_168h_true_pct}% drift</span>
        `;

        item.addEventListener('click', () => {
            selectComponent(c.component_id);
        });

        listEl.appendChild(item);
    });
}

function setupFilterButtons() {
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const cat = btn.getAttribute('data-filter');
            loadTestComponentsList(cat);
        });
    });

    // Search input
    const searchInput = document.getElementById('comp-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toUpperCase().trim();
            const filtered = allTestComponents.filter(c => c.component_id.includes(query));
            renderComponentList(filtered);
        });
    }
}

function setupPresetButtons() {
    const presetBtns = document.querySelectorAll('.preset-item-btn');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const compId = btn.getAttribute('data-id');
            selectComponent(compId);
        });
    });
}

async function selectComponent(componentId) {
    selectedComponentId = componentId;

    // Highlight selected item in list
    document.querySelectorAll('.comp-list-item').forEach(item => {
        if (item.getAttribute('data-id') === componentId) {
            item.classList.add('selected');
        } else {
            item.classList.remove('selected');
        }
    });

    // Reset result panels
    resetInferenceDisplay();

    // Fetch genuine measurements
    try {
        const res = await fetch(`/api/component/${componentId}`);
        const data = await res.json();
        currentComponentData = data;
        renderComponentData(data);
    } catch (err) {
        console.error("Error loading component:", err);
    }
}

function renderComponentData(data) {
    document.getElementById('current-comp-title').innerText = data.component_id;
    
    const gt = data.ground_truth;
    const gtBadge = document.getElementById('current-comp-gt-badge');
    if (gt.module_a_label === 0) {
        gtBadge.className = "badge badge-pass";
        gtBadge.innerText = "Ground Truth: Healthy (Class 0)";
    } else {
        gtBadge.className = "badge badge-reject";
        gtBadge.innerText = "Ground Truth: Defective (Class 1)";
    }
    
    document.getElementById('current-comp-gt-drift').innerText = `True 168h Iddq Drift: +${gt.iddq_drift_168h_true_pct}%`;

    // Populate 0h, 24h, 96h sensor values in table
    const m0 = data.measurements_0h;
    const m24 = data.measurements_24h;
    const m96 = data.measurements_96h;

    document.getElementById('val-iddq-0h').innerText = m0.iddq_uA_0h !== null ? `${m0.iddq_uA_0h.toFixed(2)} µA` : "—";
    document.getElementById('val-iddq-24h').innerText = m24.iddq_uA_24h !== null ? `${m24.iddq_uA_24h.toFixed(2)} µA` : "—";
    document.getElementById('val-iddq-96h').innerText = m96.iddq_uA_96h !== null ? `${m96.iddq_uA_96h.toFixed(2)} µA` : "—";

    document.getElementById('val-leak-0h').innerText = m0.leakage_current_uA_0h !== null ? `${m0.leakage_current_uA_0h.toFixed(2)} µA` : "—";
    document.getElementById('val-leak-24h').innerText = m24.leakage_current_uA_24h !== null ? `${m24.leakage_current_uA_24h.toFixed(2)} µA` : "—";
    document.getElementById('val-leak-96h').innerText = m96.leakage_current_uA_96h !== null ? `${m96.leakage_current_uA_96h.toFixed(2)} µA` : "—";

    document.getElementById('val-delay-0h').innerText = m0.propagation_delay_ns_0h !== null ? `${m0.propagation_delay_ns_0h.toFixed(2)} ns` : "—";
    document.getElementById('val-delay-24h').innerText = m24.propagation_delay_ns_24h !== null ? `${m24.propagation_delay_ns_24h.toFixed(2)} ns` : "—";
    document.getElementById('val-delay-96h').innerText = m96.propagation_delay_ns_96h !== null ? `${m96.propagation_delay_ns_96h.toFixed(2)} ns` : "—";

    document.getElementById('val-volt-0h').innerText = m0.voltage_V_0h !== null ? `${m0.voltage_V_0h.toFixed(3)} V` : "—";
    document.getElementById('val-volt-24h').innerText = m24.voltage_V_24h !== null ? `${m24.voltage_V_24h.toFixed(3)} V` : "—";
    document.getElementById('val-volt-96h').innerText = m96.voltage_V_96h !== null ? `${m96.voltage_V_96h.toFixed(3)} V` : "—";

    document.getElementById('val-temp-0h').innerText = m0.temperature_C_0h !== null ? `${m0.temperature_C_0h.toFixed(1)} °C` : "—";
    document.getElementById('val-temp-24h').innerText = m24.temperature_C_24h !== null ? `${m24.temperature_C_24h.toFixed(1)} °C` : "—";
    document.getElementById('val-temp-96h').innerText = m96.temperature_C_96h !== null ? `${m96.temperature_C_96h.toFixed(1)} °C` : "—";

    document.getElementById('val-drift-24h').innerText = m24.iddq_drift_24h_pct !== null ? `${m24.iddq_drift_24h_pct >= 0 ? '+' : ''}${m24.iddq_drift_24h_pct.toFixed(2)}%` : "—";
    document.getElementById('val-drift-96h').innerText = m96.iddq_drift_96h_pct !== null ? `${m96.iddq_drift_96h_pct >= 0 ? '+' : ''}${m96.iddq_drift_96h_pct.toFixed(2)}%` : "—";
}

function resetInferenceDisplay() {
    document.getElementById('inference-results-container').style.display = 'none';
    document.getElementById('sequential-timeline-container').style.display = 'none';
    document.getElementById('btn-run-96h').disabled = false;
}

/* ==============================================================================
   3. Screening Pipeline Invocations
   ============================================================================== */
function setupScreeningActionButtons() {
    const btn24 = document.getElementById('btn-run-24h');
    const btn96 = document.getElementById('btn-run-96h');
    const btnSeq = document.getElementById('btn-run-seq');

    if (btn24) {
        btn24.addEventListener('click', async () => {
            await executeScreeningGate('24h');
        });
    }

    if (btn96) {
        btn96.addEventListener('click', async () => {
            await executeScreeningGate('96h');
        });
    }

    if (btnSeq) {
        btnSeq.addEventListener('click', async () => {
            await executeSequentialScreening();
        });
    }
}

async function executeScreeningGate(gate) {
    if (!selectedComponentId) return;

    const container = document.getElementById('inference-results-container');
    container.style.display = 'block';
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    try {
        const res = await fetch(`/api/screen/${gate}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ component_id: selectedComponentId })
        });
        const data = await res.json();
        renderGateDecisionResult(data);
    } catch (err) {
        console.error("Screening execution error:", err);
    }
}

function renderGateDecisionResult(data) {
    const card = document.getElementById('decision-card');
    card.className = `decision-result-card ${data.decision}`;

    const badge = document.getElementById('decision-badge-result');
    badge.className = `badge badge-${data.decision.toLowerCase()} decision-badge-large`;
    badge.innerText = data.decision;

    document.getElementById('decision-gate-tag').innerText = `${data.screening_gate} Screening Gate`;
    document.getElementById('decision-prob-val').innerText = `${(data.defect_probability * 100).toFixed(1)}%`;
    document.getElementById('decision-drift-val').innerText = `${data.predicted_168h_iddq_drift_pct >= 0 ? '+' : ''}${data.predicted_168h_iddq_drift_pct.toFixed(2)}%`;
    document.getElementById('decision-reason-text').innerText = data.reason;
    document.getElementById('decision-recommendation-text').innerText = data.recommendation;

    // Model names
    document.getElementById('decision-model-a-name').innerText = data.model_a_name;
    document.getElementById('decision-model-b-name').innerText = data.model_b_name;
    document.getElementById('decision-features-count').innerText = `${data.num_features_used} features (Zero 168h data)`;
}

async function executeSequentialScreening() {
    if (!selectedComponentId) return;

    const seqContainer = document.getElementById('sequential-timeline-container');
    seqContainer.style.display = 'block';
    seqContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    try {
        const res = await fetch('/api/screen/sequential', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ component_id: selectedComponentId })
        });
        const data = await res.json();
        renderSequentialTimeline(data);
    } catch (err) {
        console.error("Sequential screening error:", err);
    }
}

function renderSequentialTimeline(data) {
    const s1 = data.stage_1_24h;
    const s2 = data.stage_2_96h;

    document.getElementById('seq-final-verdict').innerText = `${data.final_decision} (at ${data.final_screening_gate} Gate)`;
    document.getElementById('seq-final-summary').innerText = data.summary;

    // Stage 1 rendering
    document.getElementById('seq-s1-decision').innerText = s1.decision;
    document.getElementById('seq-s1-prob').innerText = `${(s1.defect_probability * 100).toFixed(1)}%`;
    document.getElementById('seq-s1-drift').innerText = `${s1.predicted_168h_iddq_drift_pct >= 0 ? '+' : ''}${s1.predicted_168h_iddq_drift_pct.toFixed(2)}%`;

    // Stage 2 rendering
    const s2Box = document.getElementById('seq-stage-2-box');
    if (s2) {
        s2Box.style.opacity = "1";
        document.getElementById('seq-s2-decision').innerText = s2.decision;
        document.getElementById('seq-s2-prob').innerText = `${(s2.defect_probability * 100).toFixed(1)}%`;
        document.getElementById('seq-s2-drift').innerText = `${s2.predicted_168h_iddq_drift_pct >= 0 ? '+' : ''}${s2.predicted_168h_iddq_drift_pct.toFixed(2)}%`;
    } else {
        s2Box.style.opacity = "0.45";
        document.getElementById('seq-s2-decision').innerText = "SKIPPED";
        document.getElementById('seq-s2-prob').innerText = "—";
        document.getElementById('seq-s2-drift').innerText = "—";
    }
}

/* ==============================================================================
   4. Model Performance Benchmark Loader
   ============================================================================== */
async function loadModelPerformance() {
    try {
        const res = await fetch('/api/model-performance');
        const data = await res.json();
        
        // Populate static metrics on Model Performance view if elements exist
        const a = data.module_a;
        const b = data.module_b;

        // Module A
        const a24Recall = document.getElementById('perf-a24-recall');
        if (a24Recall) a24Recall.innerText = `${(a.a24.recall * 100).toFixed(2)}%`;
        const a96Recall = document.getElementById('perf-a96-recall');
        if (a96Recall) a96Recall.innerText = `${(a.a96.recall * 100).toFixed(2)}%`;

        // Module B
        const b24R2 = document.getElementById('perf-b24-r2');
        if (b24R2) b24R2.innerText = a.b24 ? a.b24.r2_score.toFixed(4) : "0.7890";
        const b96R2 = document.getElementById('perf-b96-r2');
        if (b96R2) b96R2.innerText = b.b96 ? b.b96.r2_score.toFixed(4) : "0.9740";

    } catch (err) {
        console.error("Error loading performance data:", err);
    }
}

// -----------------------------------------------------------------------------
// Live Clock (Real-Time System Detector)
// -----------------------------------------------------------------------------
function startLiveClock() {
    const timeEl = document.getElementById('header-live-time');
    if (!timeEl) return;
    
    function updateLiveTime() {
        const now = new Date();
        const hh = String(now.getHours()).padStart(2, '0');
        const mm = String(now.getMinutes()).padStart(2, '0');
        const ss = String(now.getSeconds()).padStart(2, '0');
        timeEl.textContent = `${hh}:${mm}:${ss}`;
    }
    
    updateLiveTime();
    setInterval(updateLiveTime, 1000);
}
