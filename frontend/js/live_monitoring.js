/**
 * Cognisec AI - Live Network Monitoring Controller
 * Manages host interface selection, capture lifecycle (start/stop),
 * background telemetry polling, and real-time dual-model prediction rendering.
 */

(function () {
  'use strict';

  // Live Module State
  const LiveState = {
    interfaces: [],
    selectedInterface: null,
    isRunning: false,
    pollTimer: null,
    pollIntervalMs: 1500,
    currentFilter: 'all', // 'all' | 'suspicious' | 'benign'
    flows: [],
    status: null,
    hasInitialized: false
  };

  // DOM Elements
  const DOM = {
    interfaceSelect: document.getElementById('liveInterfaceSelect'),
    bpfFilterInput: document.getElementById('liveBpfFilterInput'),
    btnStart: document.getElementById('btnStartLiveCapture'),
    btnStop: document.getElementById('btnStopLiveCapture'),
    btnClear: document.getElementById('btnClearLiveBuffer'),
    captureDot: document.getElementById('liveCaptureDot'),
    captureStateLabel: document.getElementById('liveCaptureStateLabel'),
    captureAlert: document.getElementById('liveCaptureAlert'),

    // KPI Counters
    packetsCount: document.getElementById('livePacketsCount'),
    activeFlowsSubtext: document.getElementById('liveActiveFlowsSubtext'),
    flowsCount: document.getElementById('liveFlowsCount'),
    uptimeSubtext: document.getElementById('liveUptimeSubtext'),
    benignCount: document.getElementById('liveBenignCount'),
    benignPct: document.getElementById('liveBenignPct'),
    suspiciousCount: document.getElementById('liveSuspiciousCount'),
    suspiciousPct: document.getElementById('liveSuspiciousPct'),

    // Stream & Table
    streamCounter: document.getElementById('liveStreamCounter'),
    emptyState: document.getElementById('liveFlowsEmptyState'),
    tableContainer: document.getElementById('liveFlowsTableContainer'),
    tableBody: document.getElementById('liveFlowsTableBody'),

    // Filter Buttons
    filterAll: document.getElementById('filterAllFlows'),
    filterSuspicious: document.getElementById('filterSuspiciousFlows'),
    filterBenign: document.getElementById('filterBenignFlows'),

    // Navigation item
    liveNavItem: document.querySelector('.nav-item[data-view="live-monitoring"]')
  };

  // --------------------------------------------------------------------------
  // Utility & Helper Functions
  // --------------------------------------------------------------------------
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
  }

  function showAlert(message, type = 'info') {
    if (!DOM.captureAlert) return;
    DOM.captureAlert.style.display = 'block';
    DOM.captureAlert.className = `alert-box alert-${type}`;
    DOM.captureAlert.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
        <span>${message}</span>
        <button type="button" style="background: none; border: none; cursor: pointer; font-size: 16px; color: inherit; padding: 0 4px;" onclick="this.parentElement.parentElement.style.display='none'">&times;</button>
      </div>
    `;
  }

  function hideAlert() {
    if (DOM.captureAlert) DOM.captureAlert.style.display = 'none';
  }

  function showToastMessage(title, message, type = 'info') {
    if (typeof window.showToast === 'function') {
      window.showToast(title, message, type);
    } else {
      console.log(`[${type.toUpperCase()}] ${title}: ${message}`);
    }
  }

  // --------------------------------------------------------------------------
  // Interface Discovery
  // --------------------------------------------------------------------------
  async function loadInterfaces() {
    if (!DOM.interfaceSelect) return;
    DOM.interfaceSelect.innerHTML = '<option value="">Scanning network interfaces...</option>';

    try {
      const resp = await window.API.getLiveInterfaces();
      const ifaces = resp.interfaces || [];
      LiveState.interfaces = ifaces;

      DOM.interfaceSelect.innerHTML = '';

      if (ifaces.length === 0) {
        DOM.interfaceSelect.innerHTML = '<option value="default">Default System Interface (127.0.0.1)</option>';
        return;
      }

      let preferredId = null;

      ifaces.forEach(iface => {
        const opt = document.createElement('option');
        opt.value = iface.id;

        let tag = '';
        if (iface.is_active) tag = ' [Active]';
        else if (iface.is_loopback) tag = ' [Loopback]';

        opt.textContent = `${iface.label || iface.name}${tag}`;

        // Select Wi-Fi or active IP as first preference
        if (!preferredId && iface.is_active && !iface.is_loopback) {
          preferredId = iface.id;
        }

        DOM.interfaceSelect.appendChild(opt);
      });

      if (preferredId) {
        DOM.interfaceSelect.value = preferredId;
      } else if (ifaces.length > 0) {
        DOM.interfaceSelect.value = ifaces[0].id;
      }

    } catch (err) {
      console.error('Failed to enumerate network interfaces:', err);
      DOM.interfaceSelect.innerHTML = '<option value="default">Default System Interface (Fallback)</option>';
      showAlert(`Notice: Interface enumeration returned limited results (${err.message}). Default capture fallback is enabled.`, 'warning');
    }
  }

  // --------------------------------------------------------------------------
  // Capture Lifecycle Controls
  // --------------------------------------------------------------------------
  async function startCapture() {
    hideAlert();
    const iface = DOM.interfaceSelect ? DOM.interfaceSelect.value : null;
    const bpf = DOM.bpfFilterInput ? DOM.bpfFilterInput.value.trim() : null;

    if (DOM.btnStart) {
      DOM.btnStart.disabled = true;
      DOM.btnStart.innerHTML = '<span class="spinner" style="width:14px;height:14px;border-width:2px;margin-right:6px;"></span> Starting...';
    }

    try {
      const resp = await window.API.startLiveCapture({
        interface: iface,
        bpf_filter: bpf,
        clear_buffer: false
      });

      LiveState.isRunning = true;
      setCaptureUIState(true);
      showToastMessage('Capture Started', resp.message || 'Live packet capture initiated.', 'success');

      // Start rapid polling
      startPolling();

    } catch (err) {
      console.error('Failed to start capture:', err);
      showAlert(`Unable to start capture: ${err.message}. Ensure Npcap is installed and you are authorized to monitor this interface.`, 'warning');
      setCaptureUIState(false);
    }
  }

  async function stopCapture() {
    if (DOM.btnStop) {
      DOM.btnStop.disabled = true;
      DOM.btnStop.innerHTML = '<span class="spinner" style="width:14px;height:14px;border-width:2px;margin-right:6px;"></span> Stopping...';
    }

    try {
      const resp = await window.API.stopLiveCapture();
      LiveState.isRunning = false;
      setCaptureUIState(false);
      showToastMessage('Capture Stopped', resp.message || 'Session stopped and flows flushed.', 'info');

      // Final poll to get flushed flows
      await pollStatusAndFlows();
      stopPolling();

    } catch (err) {
      console.error('Failed to stop capture:', err);
      showAlert(`Error stopping capture session: ${err.message}`, 'warning');
      setCaptureUIState(false);
      stopPolling();
    }
  }

  async function clearBuffer() {
    if (LiveState.isRunning) {
      alert('Please stop the live capture session before clearing results.');
      return;
    }

    try {
      await window.API.clearLiveBuffer();
      LiveState.flows = [];
      updateKPIs({
        packets_seen: 0,
        active_in_flight_flows: 0,
        flows_processed: 0,
        benign_flows: 0,
        suspicious_flows: 0,
        uptime_seconds: 0
      });
      renderFlowsTable();
      renderLiveCharts({
        has_traffic: false,
        traffic_volume_over_time: [],
        vae_scores_over_time: [],
        protocol_distribution: {},
        top_suspicious_flows: [],
        telemetry_timestamp: null
      });
      showToastMessage('Buffer Cleared', 'Live flow history and counters have been reset.', 'info');
    } catch (err) {
      console.error('Failed to clear buffer:', err);
      showAlert(`Could not clear buffer: ${err.message}`, 'warning');
    }
  }

  function setCaptureUIState(running) {
    LiveState.isRunning = running;

    if (DOM.btnStart) {
      DOM.btnStart.disabled = running;
      DOM.btnStart.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        <span>${running ? 'Capturing...' : 'Start Capture'}</span>
      `;
      if (running) {
        DOM.btnStart.style.opacity = '0.5';
      } else {
        DOM.btnStart.style.opacity = '1';
      }
    }

    if (DOM.btnStop) {
      DOM.btnStop.disabled = !running;
      DOM.btnStop.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="6" width="12" height="12"/>
        </svg>
        <span>Stop Capture</span>
      `;
      if (running) {
        DOM.btnStop.style.opacity = '1';
      } else {
        DOM.btnStop.style.opacity = '0.5';
      }
    }

    if (DOM.interfaceSelect) {
      DOM.interfaceSelect.disabled = running;
    }
    if (DOM.bpfFilterInput) {
      DOM.bpfFilterInput.disabled = running;
    }

    if (DOM.captureDot) {
      if (running) {
        DOM.captureDot.className = 'status-dot capturing';
      } else {
        DOM.captureDot.className = 'status-dot';
        DOM.captureDot.style.backgroundColor = 'var(--text-light)';
        DOM.captureDot.style.animation = 'none';
      }
    }

    if (DOM.captureStateLabel) {
      if (running) {
        DOM.captureStateLabel.textContent = 'MONITORING ACTIVE';
        DOM.captureStateLabel.style.color = 'var(--status-benign)';
      } else {
        DOM.captureStateLabel.textContent = 'CAPTURE IDLE';
        DOM.captureStateLabel.style.color = 'var(--text-secondary)';
      }
    }
  }

  // --------------------------------------------------------------------------
  // Telemetry Polling Loop
  // --------------------------------------------------------------------------
  function startPolling() {
    stopPolling();
    LiveState.pollTimer = setInterval(pollStatusAndFlows, LiveState.pollIntervalMs);
  }

  function stopPolling() {
    if (LiveState.pollTimer) {
      clearInterval(LiveState.pollTimer);
      LiveState.pollTimer = null;
    }
  }

  async function pollStatusAndFlows() {
    try {
      // 1. Fetch Status
      const statusResp = await window.API.getLiveStatus();
      if (statusResp && statusResp.data) {
        const data = statusResp.data;
        LiveState.status = data;

        // Sync running state if server state diverged
        if (data.is_running !== LiveState.isRunning) {
          setCaptureUIState(data.is_running);
        }

        updateKPIs(data);

        // Show backend errors if any
        if (data.recent_errors && data.recent_errors.length > 0) {
          showAlert(`Diagnostic Notice: ${data.recent_errors[data.recent_errors.length - 1]}`, 'warning');
        }
      }

      // 2. Fetch Recent Flows
      const flowsResp = await window.API.getLiveFlows(100, 0);
      if (flowsResp && flowsResp.flows) {
        LiveState.flows = flowsResp.flows;
        renderFlowsTable();
      }

      // 3. Fetch Live Analytics & Update Dynamic SVG Telemetry Charts
      try {
        const analyticsResp = await window.API.getLiveAnalytics();
        if (analyticsResp && analyticsResp.data) {
          renderLiveCharts(analyticsResp.data);
        }
      } catch (anErr) {
        console.warn('Live analytics polling notice:', anErr);
      }

    } catch (err) {
      console.warn('Telemetry polling error:', err);
    }
  }

  // --------------------------------------------------------------------------
  // Live SVG Charts Rendering
  // --------------------------------------------------------------------------
  function renderLiveCharts(analytics) {
    if (!analytics || !window.Charts) return;

    // 1. Traffic Volume Over Time
    window.Charts.renderLiveTrafficVolume('liveTrafficVolumeChart', analytics.traffic_volume_over_time);

    // 2. VAE Anomaly Scores Over Time with calibrated threshold
    window.Charts.renderLiveVaeScores('liveVaeScoreChart', analytics.vae_scores_over_time, analytics.vae_threshold || 0.330610);

    // 3. Observed Protocol Distribution Donut
    window.Charts.renderProtocolDonut('liveProtocolChart', analytics.protocol_distribution);

    // 4. Top Suspicious Flows (Risk-Ranked with privacy IP masking)
    const topContainer = document.getElementById('liveTopSuspiciousContainer');
    if (topContainer) {
      const topFlows = analytics.top_suspicious_flows || [];
      if (topFlows.length === 0) {
        topContainer.innerHTML = '<div class="chart-empty-state"><p>No suspicious flows detected yet.</p></div>';
      } else {
        let listHtml = '<table class="data-table" style="font-size: 11.5px; margin: 0;"><thead><tr><th>Risk Rank</th><th>Endpoints (Masked)</th><th>Proto</th><th>Score</th><th>Flagged By</th></tr></thead><tbody>';
        topFlows.forEach((f, idx) => {
          listHtml += `
            <tr>
              <td><span class="badge badge-red" style="font-size: 10.5px;">#${idx + 1} Risk</span></td>
              <td><code>${f.src_ip}:${f.src_port} &rarr; ${f.dst_ip}:${f.dst_port}</code></td>
              <td><span class="badge badge-slate">${f.protocol}</span></td>
              <td><strong style="color: #dc2626;">${f.score}</strong></td>
              <td><span class="badge ${f.flagged_by === 'Both' ? 'badge-red' : (f.flagged_by === 'VAE' ? 'badge-indigo' : 'badge-blue')}">${f.flagged_by}</span></td>
            </tr>
          `;
        });
        listHtml += '</tbody></table>';
        topContainer.innerHTML = listHtml;
      }
    }

    // 5. Telemetry Timestamp
    const tsEl = document.getElementById('liveTelemetryTimestamp');
    if (tsEl) {
      if (analytics.has_traffic) {
        tsEl.textContent = `Updated: ${analytics.telemetry_timestamp || new Date().toLocaleTimeString()}`;
        tsEl.className = 'badge badge-green';
      } else if (analytics.is_capturing) {
        tsEl.textContent = 'Capture Active (Awaiting Traffic)';
        tsEl.className = 'badge badge-blue';
      } else {
        tsEl.textContent = 'Capture Idle';
        tsEl.className = 'badge badge-slate';
      }
    }
  }

  // --------------------------------------------------------------------------
  // UI Metric Updating
  // --------------------------------------------------------------------------
  function updateKPIs(status) {
    if (!status) return;

    if (DOM.packetsCount) {
      DOM.packetsCount.textContent = (status.packets_seen || 0).toLocaleString();
    }
    if (DOM.activeFlowsSubtext) {
      const active = status.active_in_flight_flows || 0;
      DOM.activeFlowsSubtext.textContent = `${active} in-flight active ${active === 1 ? 'flow' : 'flows'}`;
    }

    if (DOM.flowsCount) {
      DOM.flowsCount.textContent = (status.flows_processed || 0).toLocaleString();
    }
    if (DOM.uptimeSubtext) {
      const s = status.uptime_seconds || 0;
      DOM.uptimeSubtext.textContent = `Uptime: ${s.toFixed(1)}s`;
    }

    const total = status.flows_processed || 0;
    const benign = status.benign_flows || 0;
    const suspicious = status.suspicious_flows || 0;

    if (DOM.benignCount) {
      DOM.benignCount.textContent = benign.toLocaleString();
    }
    if (DOM.benignPct) {
      const pct = total > 0 ? ((benign / total) * 100).toFixed(1) : '0.0';
      DOM.benignPct.textContent = `${pct}% of total flows`;
    }

    if (DOM.suspiciousCount) {
      DOM.suspiciousCount.textContent = suspicious.toLocaleString();
    }
    if (DOM.suspiciousPct) {
      const pct = total > 0 ? ((suspicious / total) * 100).toFixed(1) : '0.0';
      DOM.suspiciousPct.textContent = `${pct}% flagged by models`;
    }
  }

  // --------------------------------------------------------------------------
  // Table Rendering
  // --------------------------------------------------------------------------
  function renderFlowsTable() {
    if (!DOM.tableBody) return;

    const all = LiveState.flows || [];
    let filtered = all;

    if (LiveState.currentFilter === 'suspicious') {
      filtered = all.filter(f => f.is_suspicious);
    } else if (LiveState.currentFilter === 'benign') {
      filtered = all.filter(f => !f.is_suspicious);
    }

    if (DOM.streamCounter) {
      DOM.streamCounter.textContent = `${filtered.length} of ${all.length} flows`;
    }

    if (filtered.length === 0) {
      if (DOM.emptyState) DOM.emptyState.style.display = 'block';
      if (DOM.tableContainer) DOM.tableContainer.style.display = 'none';
      DOM.tableBody.innerHTML = '';
      return;
    }

    if (DOM.emptyState) DOM.emptyState.style.display = 'none';
    if (DOM.tableContainer) DOM.tableContainer.style.display = 'block';

    const rowsHtml = filtered.map((f, idx) => {
      // Joint Decision Badge
      let badgeClass = 'badge-slate';
      let tagText = f.joint_decision || 'Normal';

      if (f.decision_level === 'critical') {
        badgeClass = 'badge-attack';
      } else if (f.decision_level === 'anomaly') {
        badgeClass = 'badge-indigo';
      } else if (f.decision_level === 'warning') {
        badgeClass = 'badge-warning';
      } else {
        badgeClass = 'badge-benign';
      }

      // XGBoost styling
      const xgb = f.xgboost || {};
      const xgbIsAtk = xgb.is_attack;
      const xgbProb = (xgb.probability !== undefined) ? (xgb.probability * 100).toFixed(1) + '%' : 'N/A';
      const xgbTag = xgbIsAtk
        ? `<span class="badge badge-attack" style="font-size: 11px;">Attack (${xgbProb})</span>`
        : `<span class="badge badge-benign" style="font-size: 11px;">Benign (${xgbProb})</span>`;

      // VAE styling
      const vae = f.vae || {};
      const vaeIsAnom = vae.is_anomaly;
      const vaeScore = (vae.anomaly_score !== undefined) ? vae.anomaly_score.toFixed(4) : 'N/A';
      const vaeTag = vaeIsAnom
        ? `<span class="badge badge-indigo" style="font-size: 11px;">Anomaly (${vaeScore})</span>`
        : `<span class="badge badge-slate" style="font-size: 11px;">Normal (${vaeScore})</span>`;

      // Endpoint formatting
      const src = `${f.src_ip}:${f.src_port}`;
      const dst = `${f.dst_ip}:${f.dst_port}`;

      return `
        <tr style="transition: background var(--transition-fast);">
          <td style="font-size: 12px; color: var(--text-muted); white-space: nowrap;">${f.timestamp || ''}</td>
          <td>
            <div style="font-family: monospace; font-size: 12.5px;">
              <span style="font-weight: 600; color: var(--text-primary);">${src}</span>
              <span style="color: var(--text-muted); margin: 0 4px;">&rarr;</span>
              <span style="font-weight: 600; color: var(--text-primary);">${dst}</span>
            </div>
          </td>
          <td>
            <span class="badge badge-slate" style="font-size: 11px; font-weight: 600;">${f.protocol || 'IP'}</span>
          </td>
          <td style="font-size: 12.5px;">${f.duration_sec ? f.duration_sec + 's' : '0.0s'}</td>
          <td style="font-size: 12px; color: var(--text-secondary); white-space: nowrap;">
            ${f.total_packets || 0} pkts / ${formatBytes(f.total_bytes)}
          </td>
          <td>${xgbTag}</td>
          <td>${vaeTag}</td>
          <td>
            <span class="badge ${badgeClass}" style="font-weight: 600; font-size: 11.5px;">
              ${tagText}
            </span>
          </td>
        </tr>
      `;
    }).join('');

    DOM.tableBody.innerHTML = rowsHtml;
  }

  // --------------------------------------------------------------------------
  // Filter Button Interactions
  // --------------------------------------------------------------------------
  function setFilter(filterType) {
    LiveState.currentFilter = filterType;

    [DOM.filterAll, DOM.filterSuspicious, DOM.filterBenign].forEach(btn => {
      if (!btn) return;
      btn.style.background = 'var(--bg-surface)';
      btn.style.color = 'var(--text-secondary)';
      btn.style.fontWeight = '500';
    });

    if (filterType === 'all' && DOM.filterAll) {
      DOM.filterAll.style.background = 'var(--brand-blue-light)';
      DOM.filterAll.style.color = 'var(--brand-blue)';
      DOM.filterAll.style.fontWeight = '600';
    } else if (filterType === 'suspicious' && DOM.filterSuspicious) {
      DOM.filterSuspicious.style.background = 'var(--status-attack-bg)';
      DOM.filterSuspicious.style.color = 'var(--status-attack)';
      DOM.filterSuspicious.style.fontWeight = '600';
    } else if (filterType === 'benign' && DOM.filterBenign) {
      DOM.filterBenign.style.background = 'var(--status-benign-bg)';
      DOM.filterBenign.style.color = 'var(--status-benign)';
      DOM.filterBenign.style.fontWeight = '600';
    }

    renderFlowsTable();
  }

  // --------------------------------------------------------------------------
  // Event Listeners & Initialization
  // --------------------------------------------------------------------------
  function init() {
    if (LiveState.hasInitialized) return;
    LiveState.hasInitialized = true;

    // Control buttons
    if (DOM.btnStart) DOM.btnStart.addEventListener('click', startCapture);
    if (DOM.btnStop) DOM.btnStop.addEventListener('click', stopCapture);
    if (DOM.btnClear) DOM.btnClear.addEventListener('click', clearBuffer);

    // Filters
    if (DOM.filterAll) DOM.filterAll.addEventListener('click', () => setFilter('all'));
    if (DOM.filterSuspicious) DOM.filterSuspicious.addEventListener('click', () => setFilter('suspicious'));
    if (DOM.filterBenign) DOM.filterBenign.addEventListener('click', () => setFilter('benign'));

    // Check status on view activation
    if (DOM.liveNavItem) {
      DOM.liveNavItem.addEventListener('click', () => {
        loadInterfaces();
        pollStatusAndFlows();
      });
    }

    // Initial interfaces load & status check
    loadInterfaces();
    pollStatusAndFlows();
  }

  // DOM ready trigger
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Export module for debugging if needed
  window.LiveMonitoring = {
    state: LiveState,
    start: startCapture,
    stop: stopCapture,
    clear: clearBuffer,
    refresh: pollStatusAndFlows
  };

})();
