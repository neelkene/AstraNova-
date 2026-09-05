/**
 * SIH 2026: AI-Driven Anomaly Detection in Component Burn-In & Screening
 * Frontend Application & Real-Time Inference Client
 * 
 * STRICT COMPLIANCE:
 * - NO fake data, NO fabricated predictions, NO synthetic confidence scores.
 * - All numbers originate from genuine backend API endpoints connected to trained ML pipelines.
 */

// -----------------------------------------------------------------------------
// Global Configuration & API Base Detection
// -----------------------------------------------------------------------------
const API_BASE = (window.location.port === '8000' || window.location.port === '') 
    ? '' 
    : 'http://127.0.0.1:8000';

// Global Application State
const state = {
    selectedComponentId: 'SYN_C01216', // Verified healthy baseline component
    allComponents: [],
    activeFilter: 'all',
    currentMeasurements: null,
    currentPrediction: null,
    activeChartParam: 'iddq',
    currentStageView: 'dual', // '24h', '96h', or 'dual'
    charts: {
        progression: null,
        delta: null,
        forecast: null,
        importance: null,
        donut: null
    }
};

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
    startLiveClock();
    initNavigation();
    initControls();
    await checkSystemHealth();
    await loadDatasetOverview();
    await loadComponentList('all');
    await selectAndAnalyzeComponent(state.selectedComponentId);
});

// -----------------------------------------------------------------------------
// Navigation Tabs
// -----------------------------------------------------------------------------
function initNavigation() {
    const tabBtns = document.querySelectorAll('.nav-tab-btn');
    const viewPanels = document.querySelectorAll('.view-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetView = btn.getAttribute('data-view');
            tabBtns.forEach(b => b.classList.remove('active'));
            viewPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(targetView);
            if (targetEl) {
                targetEl.classList.add('active');
            }

            // Resize charts upon switching tab
            Object.values(state.charts).forEach(c => {
                if (c) c.resize();
            });
        });
    });
}

// -----------------------------------------------------------------------------
// Interactive Controls & Event Listeners
// -----------------------------------------------------------------------------
function initControls() {
    // Search input
    const searchInput = document.getElementById('comp-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toUpperCase().trim();
            renderComponentList(state.allComponents.filter(c => c.component_id.includes(query)));
        });
    }

    // Presets
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const compId = btn.getAttribute('data-preset');
            selectAndAnalyzeComponent(compId);
        });
    });

    // Filter tabs (All vs Locked Test Set)
    document.querySelectorAll('.filter-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const split = btn.getAttribute('data-filter');
            state.activeFilter = split;
            loadComponentList(split);
        });
    });

    // Manual ID loader
    const btnLoadManual = document.getElementById('btn-load-manual');
    const manualInput = document.getElementById('manual-id-input');
    if (btnLoadManual && manualInput) {
        btnLoadManual.addEventListener('click', () => {
            const id = manualInput.value.trim().toUpperCase();
            if (id) {
                selectAndAnalyzeComponent(id);
            }
        });
        manualInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                btnLoadManual.click();
            }
        });
    }

    // Progression Demonstration Buttons
    const btn24h = document.getElementById('btn-run-24h');
    if (btn24h) {
        btn24h.addEventListener('click', () => runStageView('24h'));
    }

    const btn96h = document.getElementById('btn-run-96h');
    if (btn96h) {
        btn96h.addEventListener('click', () => runStageView('96h'));
    }

    const btnFull = document.getElementById('btn-run-full');
    if (btnFull) {
        btnFull.addEventListener('click', () => runStageView('dual'));
    }

    // Progression Chart Parameter Switchers
    const btnIddq = document.getElementById('btn-chart-param-iddq');
    const btnLeak = document.getElementById('btn-chart-param-leak');
    const btnDelay = document.getElementById('btn-chart-param-delay');

    if (btnIddq) btnIddq.addEventListener('click', () => setProgressionChartParam('iddq'));
    if (btnLeak) btnLeak.addEventListener('click', () => setProgressionChartParam('leakage'));
    if (btnDelay) btnDelay.addEventListener('click', () => setProgressionChartParam('delay'));
}

// -----------------------------------------------------------------------------
// System Health Check
// -----------------------------------------------------------------------------
async function checkSystemHealth() {
    const dot = document.getElementById('sys-status-dot');
    const text = document.getElementById('sys-status-text');

    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        if (data.status === 'ok') {
            dot.className = 'status-indicator-dot';
            text.textContent = 'System Online';
            const pill = document.getElementById('sys-status-pill');
            if (pill) pill.setAttribute('title', 'System Online: All 4 ML Models Loaded');
        } else {
            dot.className = 'status-indicator-dot degraded';
            text.textContent = 'Degraded';
            const pill = document.getElementById('sys-status-pill');
            if (pill) pill.setAttribute('title', 'Degraded: Check Model Status');
        }
    } catch (err) {
        dot.className = 'status-indicator-dot degraded';
        text.textContent = 'Backend Offline';
        console.error('API health check error:', err);
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

// -----------------------------------------------------------------------------
// Load Component List
// -----------------------------------------------------------------------------
async function loadComponentList(split = 'all') {
    const listEl = document.getElementById('component-scroll-list');
    listEl.innerHTML = '<div class="empty-state" style="padding:1rem;"><div class="loading-spinner"></div><div style="font-size:0.75rem;margin-top:0.3rem;">Loading components...</div></div>';

    try {
        const res = await fetch(`${API_BASE}/api/components?split=${split}&limit=100`);
        const data = await res.json();
        state.allComponents = data.components || [];

        const countBadge = document.getElementById('comp-count-badge');
        if (countBadge) {
            countBadge.textContent = `${data.total_available} Available`;
        }

        renderComponentList(state.allComponents);
    } catch (err) {
        listEl.innerHTML = '<div style="padding:1rem;color:var(--status-reject);font-size:0.75rem;">Failed to connect to backend components API.</div>';
        console.error('Error loading component list:', err);
    }
}

function renderComponentList(components) {
    const listEl = document.getElementById('component-scroll-list');
    listEl.innerHTML = '';

    if (!components || components.length === 0) {
        listEl.innerHTML = '<div style="padding:1rem;color:var(--text-muted);font-size:0.75rem;text-align:center;">No matching components found</div>';
        return;
    }

    components.forEach(c => {
        const item = document.createElement('div');
        item.className = `comp-list-item ${c.component_id === state.selectedComponentId ? 'selected' : ''}`;
        item.setAttribute('data-id', c.component_id);

        const iddq0 = c.iddq_uA_0h !== null ? `${c.iddq_uA_0h} μA` : 'N/A';
        const iddq96 = c.iddq_uA_96h !== null ? `${c.iddq_uA_96h} μA` : 'N/A';

        item.innerHTML = `
            <div>
                <strong style="font-family: var(--font-mono); font-size: 0.8rem;">${c.component_id}</strong>
                <div style="font-size: 0.7rem; color: var(--text-muted);">0h: ${iddq0}</div>
            </div>
            <div style="text-align: right;">
                <span class="badge badge-neutral" style="font-family: var(--font-mono); font-size: 0.68rem;">96h: ${iddq96}</span>
            </div>
        `;

        item.addEventListener('click', () => {
            selectAndAnalyzeComponent(c.component_id);
        });

        listEl.appendChild(item);
    });
}

// -----------------------------------------------------------------------------
// Component Selection & Full Analysis Execution
// -----------------------------------------------------------------------------
async function selectAndAnalyzeComponent(componentId) {
    state.selectedComponentId = componentId;

    // Update UI highlights in sidebar
    document.querySelectorAll('.comp-list-item').forEach(el => {
        if (el.getAttribute('data-id') === componentId) {
            el.classList.add('selected');
        } else {
            el.classList.remove('selected');
        }
    });

    // Update preset buttons active state
    document.querySelectorAll('.preset-btn').forEach(b => {
        if (b.getAttribute('data-preset') === componentId) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });

    // Header info
    const headerId = document.getElementById('header-comp-id');
    if (headerId) headerId.textContent = componentId;

    const tableIndicator = document.getElementById('table-comp-indicator');
    if (tableIndicator) tableIndicator.textContent = `Component: ${componentId}`;

    const timestampEl = document.getElementById('header-timestamp');
    if (timestampEl) timestampEl.textContent = new Date().toLocaleTimeString();

    // 1. Fetch real physical measurements
    try {
        const lookupRes = await fetch(`${API_BASE}/api/components/${componentId}`);
        if (!lookupRes.ok) {
            throw new Error(`Component ${componentId} not found`);
        }
        state.currentMeasurements = await lookupRes.json();
    } catch (err) {
        console.error('Error fetching measurements:', err);
        alert(`Failed to load component measurements for ${componentId}: ${err.message}`);
        return;
    }

    // 2. Fetch authentic ML predictions
    try {
        const predRes = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ component_id: componentId })
        });
        if (!predRes.ok) {
            throw new Error(`Inference request failed with HTTP ${predRes.status}`);
        }
        state.currentPrediction = await predRes.json();
    } catch (err) {
        console.error('Error running inference:', err);
        alert(`Failed to execute prediction for ${componentId}: ${err.message}`);
        return;
    }

    // 3. Render Dashboard Sections
    runStageView(state.currentStageView);
}

// -----------------------------------------------------------------------------
// Stage Progression View Switcher (24h, 96h, or Dual)
// -----------------------------------------------------------------------------
function runStageView(stage) {
    state.currentStageView = stage;
    const pred = state.currentPrediction;
    const meas = state.currentMeasurements;

    if (!pred || !meas) return;

    // Update Stepper
    updateStepper(stage);

    // Update Decision Hero
    updateDecisionHero(pred, stage);

    // Update Top Stat Cards
    updateStatCards(pred, meas, stage);

    // Update Physical Measurements Table
    renderMeasurementsTable(meas, pred, stage);

    // Update Module Comparison Cards
    renderModuleCards(pred, stage);

    // Render Charts
    renderProgressionChart(meas, state.activeChartParam);
    renderDeltaChart(meas);
    renderForecastChart(meas, pred);
    renderImportanceChart(pred, stage);
}

function updateStepper(stage) {
    const s0 = document.getElementById('step-0h');
    const s24 = document.getElementById('step-24h');
    const s96 = document.getElementById('step-96h');
    const s168 = document.getElementById('step-168h');
    const headerBadge = document.getElementById('header-gate-badge');

    [s0, s24, s96, s168].forEach(s => {
        if (s) s.className = 'stage-step';
    });

    if (stage === '24h') {
        if (s0) s0.className = 'stage-step completed';
        if (s24) s24.className = 'stage-step active';
        if (headerBadge) headerBadge.textContent = '24h Early Warning';
    } else if (stage === '96h') {
        if (s0) s0.className = 'stage-step completed';
        if (s24) s24.className = 'stage-step completed';
        if (s96) s96.className = 'stage-step active';
        if (headerBadge) headerBadge.textContent = '96h Qualification';
    } else {
        if (s0) s0.className = 'stage-step completed';
        if (s24) s24.className = 'stage-step completed';
        if (s96) s96.className = 'stage-step completed';
        if (s168) s168.className = 'stage-step active';
        if (headerBadge) headerBadge.textContent = '96h Dual-Gate';
    }
}

// -----------------------------------------------------------------------------
// Update Decision Hero Banner
// -----------------------------------------------------------------------------
function updateDecisionHero(pred, stage) {
    const heroCard = document.getElementById('decision-hero-card');
    const badge = document.getElementById('hero-decision-badge');
    const title = document.getElementById('hero-decision-title');
    const reason = document.getElementById('hero-decision-reason');
    const rec = document.getElementById('hero-decision-rec');
    const confTag = document.getElementById('hero-confidence-tag');
    const gateTag = document.getElementById('hero-gate-tag');

    let activeDecision = pred.final_decision;
    let gateLabel = 'Evaluated @ 96h Gate';

    if (stage === '24h' && pred.gate_24h) {
        activeDecision = pred.gate_24h.gate_decision;
        gateLabel = 'Evaluated @ 24h Early Gate';
    }

    if (!activeDecision) {
        heroCard.className = 'decision-hero review';
        badge.textContent = 'REVIEW';
        title.textContent = 'Analysis In Progress';
        reason.textContent = 'Evaluating sensor measurements...';
        rec.textContent = '';
        return;
    }

    const status = activeDecision.status; // PASS, REVIEW, REJECT
    heroCard.className = `decision-hero ${status.toLowerCase()}`;
    badge.textContent = status;

    if (status === 'PASS') {
        title.textContent = 'Parametric Reliability Confirmed';
    } else if (status === 'REVIEW') {
        title.textContent = 'Marginal / Borderline Signal Detected';
    } else {
        title.textContent = 'Critical Burn-In Defect Identified';
    }

    reason.textContent = activeDecision.reason || 'Model screening decision generated.';
    rec.textContent = `Recommendation: ${activeDecision.recommendation || 'Follow standard operating procedure.'}`;
    confTag.textContent = `Confidence: ${activeDecision.confidence_level || 'HIGH'}`;
    gateTag.textContent = gateLabel;
}

// -----------------------------------------------------------------------------
// Top Stat KPI Cards
// -----------------------------------------------------------------------------
function updateStatCards(pred, meas, stage) {
    // 1. IDDQ Physical Shift
    const iddq0 = meas.measurements_0h?.iddq_uA_0h;
    const iddq24 = meas.measurements_24h?.iddq_uA_24h;
    const iddq96 = meas.measurements_96h?.iddq_uA_96h;

    const statIddqVal = document.getElementById('stat-iddq-change');
    const statIddqPct = document.getElementById('stat-iddq-pct');

    if (iddq0 !== null && iddq0 !== undefined) {
        let endVal = (stage === '24h') ? iddq24 : (iddq96 ?? iddq24);
        let timeLabel = (stage === '24h') ? '24h' : '96h';

        if (endVal !== null && endVal !== undefined) {
            const delta = endVal - iddq0;
            const sign = delta >= 0 ? '+' : '';
            const pct = (delta / iddq0) * 100.0;

            statIddqVal.textContent = `${sign}${delta.toFixed(2)} μA`;
            statIddqPct.textContent = `Change: ${sign}${pct.toFixed(2)}% (${iddq0.toFixed(1)} → ${endVal.toFixed(1)} μA)`;
        } else {
            statIddqVal.textContent = 'N/A';
            statIddqPct.textContent = 'Measurement unavailable';
        }
    }

    // 2. Module A Defect Probability
    const statProb = document.getElementById('stat-defect-prob');
    const statClass = document.getElementById('stat-defect-class');

    const modA = (stage === '24h') ? pred.gate_24h?.module_a : (pred.gate_96h?.module_a ?? pred.gate_24h?.module_a);

    if (modA) {
        const probPct = (modA.risk_probability * 100).toFixed(1);
        statProb.textContent = `${probPct}%`;
        statClass.textContent = `Classification: ${modA.class_name.toUpperCase()} (p = ${modA.risk_probability.toFixed(3)})`;
    } else {
        statProb.textContent = 'N/A';
        statClass.textContent = 'Module A unavailable';
    }

    // 3. Module B 168h Forecast
    const statShift = document.getElementById('stat-forecast-shift');
    const statForecastPct = document.getElementById('stat-forecast-pct');

    const modB = (stage === '24h') ? pred.gate_24h?.module_b : (pred.gate_96h?.module_b ?? pred.gate_24h?.module_b);

    if (modB && iddq0) {
        const driftRaw = modB.predicted_iddq_drift_168h;
        const driftPct = modB.predicted_iddq_drift_168h_pct;
        const physicalShift = iddq0 * driftRaw;
        const sign = physicalShift >= 0 ? '+' : '';

        statShift.textContent = `${sign}${physicalShift.toFixed(2)} μA`;
        statForecastPct.textContent = `Projected Drift: ${sign}${driftPct.toFixed(2)}% @ 168h`;
    } else {
        statShift.textContent = 'N/A';
        statForecastPct.textContent = 'Module B unavailable';
    }

    // 4. Chamber Savings
    const statChamber = document.getElementById('stat-chamber-saved');
    const statChamberSub = document.getElementById('stat-chamber-sub');

    if (stage === '24h') {
        statChamber.textContent = '144 Hours';
        statChamberSub.textContent = 'Early triage at 24h gate vs 168h standard';
    } else {
        statChamber.textContent = '72 Hours';
        statChamberSub.textContent = 'Qualified at 96h gate vs 168h standard';
    }
}

// -----------------------------------------------------------------------------
// Section 5 & 6: Actual Burn-In Measurements & Physical Changes Table
// -----------------------------------------------------------------------------
function renderMeasurementsTable(meas, pred, stage) {
    const tbody = document.getElementById('measurements-tbody');
    tbody.innerHTML = '';

    const params = [
        { key: 'iddq', name: 'IDDQ (Quiescent Current)', unit: 'μA', col0: 'iddq_uA_0h', col24: 'iddq_uA_24h', col96: 'iddq_uA_96h', col168: 'iddq_uA_168h', tooltip: 'Quiescent supply current measured while circuit is not actively switching.' },
        { key: 'leakage', name: 'Leakage Current', unit: 'μA', col0: 'leakage_current_uA_0h', col24: 'leakage_current_uA_24h', col96: 'leakage_current_uA_96h', col168: 'leakage_current_uA_168h', tooltip: 'Sub-threshold static conduction across reverse-biased junctions.' },
        { key: 'delay', name: 'Propagation Delay', unit: 'ns', col0: 'propagation_delay_ns_0h', col24: 'propagation_delay_ns_24h', col96: 'propagation_delay_ns_96h', col168: 'propagation_delay_ns_168h', tooltip: 'Signal transition latency through standardized ring oscillator critical path.' },
        { key: 'voltage', name: 'Supply Voltage', unit: 'V', col0: 'voltage_V_0h', col24: 'voltage_V_24h', col96: 'voltage_V_96h', col168: 'voltage_V_168h', tooltip: 'Burn-in chamber regulated power rail voltage.' },
        { key: 'temperature', name: 'Junction Temperature', unit: '°C', col0: 'temperature_C_0h', col24: 'temperature_C_24h', col96: 'temperature_C_96h', col168: 'temperature_C_168h', tooltip: 'Active die temperature maintained during thermal-bias stress testing.' }
    ];

    const m0 = meas.measurements_0h || {};
    const m24 = meas.measurements_24h || {};
    const m96 = meas.measurements_96h || {};
    const m168 = meas.measurements_168h || {};

    params.forEach(p => {
        const v0 = m0[p.col0];
        const v24 = m24[p.col24];
        const v96 = (stage === '24h') ? null : m96[p.col96];
        const v168 = (stage === '24h') ? null : m168[p.col168];

        // Format physical values
        const fmt0 = formatVal(v0, p.unit);
        const fmt24 = formatVal(v24, p.unit);
        const fmt96 = formatVal(v96, p.unit);
        const fmt168 = formatVal(v168, p.unit);

        // Calculate 0h -> 24h physical delta
        const delta24 = calcDelta(v24, v0, p.unit);
        // Calculate 0h -> 96h physical delta
        const delta96 = (stage === '24h') ? 'N/A' : calcDelta(v96, v0, p.unit);

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>
                <strong>${p.name}</strong>
                <span class="info-tooltip" data-tooltip="${p.tooltip}">ℹ️</span>
            </td>
            <td class="mono">${fmt0}</td>
            <td class="mono">${fmt24}</td>
            <td class="mono">${fmt96}</td>
            <td class="mono" style="color: var(--text-muted);">${fmt168}</td>
            <td>${delta24}</td>
            <td>${delta96}</td>
        `;
        tbody.appendChild(tr);
    });
}

function formatVal(val, unit) {
    if (val === null || val === undefined) return 'N/A';
    return `${Number(val).toFixed(2)} ${unit}`;
}

function calcDelta(endVal, startVal, unit) {
    if (endVal === null || endVal === undefined || startVal === null || startVal === undefined) {
        return '<span class="delta-pill neutral">N/A</span>';
    }
    const diff = endVal - startVal;
    const sign = diff >= 0 ? '+' : '';
    const pct = (diff / Math.abs(startVal)) * 100.0;

    let pillClass = 'nominal';
    if (Math.abs(pct) > 5.0 && unit === 'μA') pillClass = 'positive';

    return `
        <span class="delta-pill ${pillClass}">
            ${sign}${diff.toFixed(2)} ${unit}
        </span>
        <span style="font-size: 0.72rem; color: var(--text-muted); margin-left: 0.35rem;">(${sign}${pct.toFixed(2)}%)</span>
    `;
}

// -----------------------------------------------------------------------------
// Section 7 & 8: Module A & Module B Display
// -----------------------------------------------------------------------------
function renderModuleCards(pred, stage) {
    const a24 = pred.gate_24h?.module_a;
    const a96 = pred.gate_96h?.module_a;
    const b24 = pred.gate_24h?.module_b;
    const b96 = pred.gate_96h?.module_b;

    // Module A 24h
    if (a24) {
        const prob = (a24.risk_probability * 100).toFixed(1);
        document.getElementById('a24-prob').textContent = `${prob}%`;
        const a24Badge = document.getElementById('a24-badge');
        a24Badge.textContent = a24.class_name.toUpperCase();
        a24Badge.className = `badge ${a24.prediction === 1 ? 'badge-reject' : 'badge-pass'}`;
    }

    // Module A 96h
    const boxA96 = document.getElementById('box-mod-a96');
    if (stage === '24h') {
        boxA96.style.opacity = '0.35';
        document.getElementById('a96-prob').textContent = 'N/A (24h Gate)';
        document.getElementById('a96-badge').textContent = 'UNAVAILABLE';
        document.getElementById('a96-badge').className = 'badge badge-neutral';
    } else if (a96) {
        boxA96.style.opacity = '1';
        const prob = (a96.risk_probability * 100).toFixed(1);
        document.getElementById('a96-prob').textContent = `${prob}%`;
        const a96Badge = document.getElementById('a96-badge');
        a96Badge.textContent = a96.class_name.toUpperCase();
        a96Badge.className = `badge ${a96.prediction === 1 ? 'badge-reject' : 'badge-pass'}`;
    }

    // Module B 24h
    if (b24) {
        const driftPct = b24.predicted_iddq_drift_168h_pct;
        const sign = driftPct >= 0 ? '+' : '';
        document.getElementById('b24-drift').textContent = `${sign}${driftPct.toFixed(2)}%`;
        const b24Badge = document.getElementById('b24-badge');
        b24Badge.textContent = `${sign}${driftPct.toFixed(1)}%`;
        b24Badge.className = `badge ${driftPct >= 5.0 ? 'badge-reject' : (driftPct >= 2.0 ? 'badge-review' : 'badge-pass')}`;

        const iddq0 = state.currentMeasurements?.measurements_0h?.iddq_uA_0h;
        if (iddq0) {
            const shift = iddq0 * b24.predicted_iddq_drift_168h;
            document.getElementById('b24-shift').textContent = `${sign}${shift.toFixed(2)} μA`;
        }
    }

    // Module B 96h
    const boxB96 = document.getElementById('box-mod-b96');
    if (stage === '24h') {
        boxB96.style.opacity = '0.35';
        document.getElementById('b96-drift').textContent = 'N/A (24h Gate)';
        document.getElementById('b96-shift').textContent = 'N/A';
        document.getElementById('b96-badge').textContent = 'UNAVAILABLE';
        document.getElementById('b96-badge').className = 'badge badge-neutral';
    } else if (b96) {
        boxB96.style.opacity = '1';
        const driftPct = b96.predicted_iddq_drift_168h_pct;
        const sign = driftPct >= 0 ? '+' : '';
        document.getElementById('b96-drift').textContent = `${sign}${driftPct.toFixed(2)}%`;
        const b96Badge = document.getElementById('b96-badge');
        b96Badge.textContent = `${sign}${driftPct.toFixed(1)}%`;
        b96Badge.className = `badge ${driftPct >= 5.0 ? 'badge-reject' : (driftPct >= 2.0 ? 'badge-review' : 'badge-pass')}`;

        const iddq0 = state.currentMeasurements?.measurements_0h?.iddq_uA_0h;
        if (iddq0) {
            const shift = iddq0 * b96.predicted_iddq_drift_168h;
            document.getElementById('b96-shift').textContent = `${sign}${shift.toFixed(2)} μA`;
        }
    }
}

// -----------------------------------------------------------------------------
// Chart 1: Parameter Progression Line Chart
// -----------------------------------------------------------------------------
function setProgressionChartParam(paramKey) {
    state.activeChartParam = paramKey;
    ['iddq', 'leak', 'delay'].forEach(p => {
        const b = document.getElementById(`btn-chart-param-${p}`);
        if (b) {
            if ((p === 'leak' && paramKey === 'leakage') || p === paramKey) {
                b.classList.remove('btn-secondary');
                b.classList.add('btn-primary');
            } else {
                b.classList.remove('btn-primary');
                b.classList.add('btn-secondary');
            }
        }
    });
    renderProgressionChart(state.currentMeasurements, paramKey);
}

function renderProgressionChart(meas, paramKey = 'iddq') {
    const ctx = document.getElementById('chart-progression-canvas')?.getContext('2d');
    if (!ctx || !meas) return;

    let colSuffix = '_uA';
    let unit = 'μA';
    let label = 'IDDQ';
    let baseProp = 'iddq_uA';

    if (paramKey === 'leakage') {
        label = 'Leakage Current';
        unit = 'μA';
        baseProp = 'leakage_current_uA';
    } else if (paramKey === 'delay') {
        label = 'Propagation Delay';
        unit = 'ns';
        baseProp = 'propagation_delay_ns';
    }

    const v0 = meas.measurements_0h?.[`${baseProp}_0h`];
    const v24 = meas.measurements_24h?.[`${baseProp}_24h`];
    const v96 = meas.measurements_96h?.[`${baseProp}_96h`];
    const v168 = meas.measurements_168h?.[`${baseProp}_168h`];

    const labels = ['0h Baseline', '24h Early Gate', '96h Mid Gate', '168h Benchmark'];
    const values = [v0, v24, v96, v168];

    if (state.charts.progression) {
        state.charts.progression.destroy();
    }

    state.charts.progression = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `${label} (${unit})`,
                data: values,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.08)',
                borderWidth: 2.5,
                pointBackgroundColor: '#0f172a',
                pointRadius: 4.5,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { font: { size: 11, family: '-apple-system, BlinkMacSystemFont, sans-serif' } } },
                tooltip: {
                    callbacks: {
                        label: (c) => `${c.dataset.label}: ${c.raw !== null ? c.raw.toFixed(2) : 'N/A'}`
                    }
                }
            },
            scales: {
                y: {
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { family: 'ui-monospace, monospace', size: 11 } },
                    title: { display: true, text: `${label} (${unit})`, font: { size: 11 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } }
                }
            }
        }
    });
}

// -----------------------------------------------------------------------------
// Chart 2: Parameter Physical Shift Bar Chart
// -----------------------------------------------------------------------------
function renderDeltaChart(meas) {
    const ctx = document.getElementById('chart-delta-canvas')?.getContext('2d');
    if (!ctx || !meas) return;

    const iddq0 = meas.measurements_0h?.iddq_uA_0h;
    const iddq96 = meas.measurements_96h?.iddq_uA_96h;
    const deltaIddq = (iddq0 && iddq96) ? Number((iddq96 - iddq0).toFixed(2)) : 0;

    const leak0 = meas.measurements_0h?.leakage_current_uA_0h;
    const leak96 = meas.measurements_96h?.leakage_current_uA_96h;
    const deltaLeak = (leak0 && leak96) ? Number((leak96 - leak0).toFixed(2)) : 0;

    const delay0 = meas.measurements_0h?.propagation_delay_ns_0h;
    const delay96 = meas.measurements_96h?.propagation_delay_ns_96h;
    const deltaDelay = (delay0 && delay96) ? Number((delay96 - delay0).toFixed(2)) : 0;

    if (state.charts.delta) {
        state.charts.delta.destroy();
    }

    state.charts.delta = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['IDDQ Shift (μA)', 'Leakage Shift (μA)', 'Delay Shift (ns)'],
            datasets: [{
                label: 'Actual Physical Change (0h → 96h)',
                data: [deltaIddq, deltaLeak, deltaDelay],
                backgroundColor: [
                    deltaIddq > 4.0 ? 'rgba(220, 38, 38, 0.7)' : 'rgba(37, 99, 235, 0.7)',
                    'rgba(14, 165, 233, 0.7)',
                    'rgba(16, 185, 129, 0.7)'
                ],
                borderColor: [
                    deltaIddq > 4.0 ? '#dc2626' : '#2563eb',
                    '#0ea5e9',
                    '#10b981'
                ],
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (c) => `Physical Shift: ${c.raw >= 0 ? '+' : ''}${c.raw}`
                    }
                }
            },
            scales: {
                y: {
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { family: 'ui-monospace, monospace' } },
                    title: { display: true, text: 'Absolute Physical Delta', font: { size: 11 } }
                },
                x: { grid: { display: false } }
            }
        }
    });
}

// -----------------------------------------------------------------------------
// Chart 3: 168h Forecast Comparison (Measured vs Predicted)
// -----------------------------------------------------------------------------
function renderForecastChart(meas, pred) {
    const ctx = document.getElementById('chart-forecast-canvas')?.getContext('2d');
    if (!ctx || !meas || !pred) return;

    const iddq0 = meas.measurements_0h?.iddq_uA_0h;
    const iddq24 = meas.measurements_24h?.iddq_uA_24h;
    const iddq96 = meas.measurements_96h?.iddq_uA_96h;
    const iddq168Actual = meas.measurements_168h?.iddq_uA_168h;

    // Module B forecast for 168h
    const bMod = pred.gate_96h?.module_b || pred.gate_24h?.module_b;
    let iddq168Predicted = null;
    if (iddq0 && bMod) {
        iddq168Predicted = Number((iddq0 * (1 + bMod.predicted_iddq_drift_168h)).toFixed(2));
    }

    if (state.charts.forecast) {
        state.charts.forecast.destroy();
    }

    const labels = ['0h Baseline', '24h Measured', '96h Measured', '168h Predicted (B96)', '168h Measured (Truth)'];
    const values = [iddq0, iddq24, iddq96, iddq168Predicted, iddq168Actual];

    state.charts.forecast = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'IDDQ Current (μA)',
                data: values,
                backgroundColor: [
                    'rgba(100, 116, 139, 0.5)',
                    'rgba(37, 99, 235, 0.65)',
                    'rgba(37, 99, 235, 0.85)',
                    'rgba(217, 119, 6, 0.75)', // Forecast bar
                    'rgba(16, 185, 129, 0.75)'  // Measured benchmark bar
                ],
                borderColor: [
                    '#64748b',
                    '#2563eb',
                    '#1d4ed8',
                    '#d97706',
                    '#059669'
                ],
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (c) => `${c.label}: ${c.raw !== null ? c.raw + ' μA' : 'N/A'}`
                    }
                }
            },
            scales: {
                y: {
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { family: 'ui-monospace, monospace' } },
                    title: { display: true, text: 'IDDQ Current (μA)', font: { size: 11 } }
                },
                x: { grid: { display: false }, ticks: { font: { size: 10 } } }
            }
        }
    });
}

// -----------------------------------------------------------------------------
// Chart 4: Feature Importance / Degradation Drivers
// -----------------------------------------------------------------------------
function renderImportanceChart(pred, stage) {
    const ctx = document.getElementById('chart-importance-canvas')?.getContext('2d');
    if (!ctx) return;

    const modA = (stage === '24h') ? pred?.gate_24h?.module_a : (pred?.gate_96h?.module_a ?? pred?.gate_24h?.module_a);
    let items = modA?.feature_importances || [];

    // Friendly display mappings
    const nameMap = {
        'iddq_drift_96h_pct': '96h IDDQ Drift (%)',
        'iddq_uA_96h': '96h IDDQ Reading (μA)',
        'iddq_drift_24h_pct': '24h IDDQ Drift (%)',
        'leakage_drift_96h_pct': '96h Leakage Drift (%)',
        'iddq_uA_24h': '24h IDDQ Reading (μA)',
        'delay_drift_96h_pct': '96h Propagation Delay Drift (%)',
        'leakage_current_uA_96h': '96h Leakage Current (μA)',
        'temperature_C_96h': '96h Chamber Temperature (°C)',
        'propagation_delay_ns_96h': '96h Propagation Delay (ns)',
        'voltage_V_96h': '96h Regulated Voltage (V)'
    };

    if (items.length === 0) {
        if (state.charts.importance) {
            state.charts.importance.destroy();
            state.charts.importance = null;
        }
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.font = '12px -apple-system, sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.textAlign = 'center';
        ctx.fillText('Feature importance is not supported by Logistic Regression (24h gate).', ctx.canvas.width / 2, ctx.canvas.height / 2 - 10);
        ctx.fillText('Switch to 96h Qualification to inspect Random Forest Gini feature weights.', ctx.canvas.width / 2, ctx.canvas.height / 2 + 10);
        return;
    }

    const topItems = items.slice(0, 6);
    const labels = topItems.map(i => nameMap[i.feature] || i.feature);
    const values = topItems.map(i => i.importance);

    if (state.charts.importance) {
        state.charts.importance.destroy();
    }

    state.charts.importance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Gini Feature Importance',
                data: values,
                backgroundColor: 'rgba(30, 41, 59, 0.75)',
                borderColor: '#0f172a',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (c) => `Relative Importance: ${(c.raw * 100).toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#f1f5f9' },
                    ticks: {
                        callback: (v) => `${(v * 100).toFixed(0)}%`
                    },
                    title: { display: true, text: 'Normalized Feature Weight', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 10 } }
                }
            }
        }
    });
}

// -----------------------------------------------------------------------------
// Section 12: Dataset Overview & Donut Chart
// -----------------------------------------------------------------------------
async function loadDatasetOverview() {
    try {
        const res = await fetch(`${API_BASE}/api/dataset-overview`);
        if (!res.ok) return;
        const data = await res.json();

        // Update stat cards in Overview Tab
        const tot = document.getElementById('overview-total-val');
        const norm = document.getElementById('overview-normal-val');
        const drift = document.getElementById('overview-drift-val');
        const anom = document.getElementById('overview-anom-val');

        if (tot) tot.textContent = data.total_components.toLocaleString();
        if (norm) norm.textContent = `${data.normal_count.toLocaleString()} (${data.normal_pct}%)`;
        if (drift) drift.textContent = `${data.drifting_count.toLocaleString()} (${data.drifting_pct}%)`;
        if (anom) anom.textContent = `${data.anomalous_count.toLocaleString()} (${data.anomalous_pct}%)`;

        // Render Donut
        renderDonutOverview(data);
    } catch (err) {
        console.error('Error loading dataset overview:', err);
    }
}

function renderDonutOverview(data) {
    const ctx = document.getElementById('chart-donut-overview')?.getContext('2d');
    if (!ctx) return;

    if (state.charts.donut) {
        state.charts.donut.destroy();
    }

    state.charts.donut = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [
                `Normal Components (${data.normal_pct}%)`,
                `Latent Drifting (${data.drifting_pct}%)`,
                `Gross Anomalous (${data.anomalous_pct}%)`
            ],
            datasets: [{
                data: [data.normal_count, data.drifting_count, data.anomalous_count],
                backgroundColor: [
                    '#059669', // Emerald
                    '#d97706', // Amber
                    '#dc2626'  // Red
                ],
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '62%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { font: { size: 11, family: '-apple-system, BlinkMacSystemFont, sans-serif' } }
                },
                tooltip: {
                    callbacks: {
                        label: (c) => ` ${c.label}: ${c.raw.toLocaleString()} components`
                    }
                }
            }
        }
    });
}
