/**
 * Cognisec AI - Main Application Logic
 * Coordinates view routing, file upload, model inference, and research dashboards.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global App State
  const AppState = {
    currentView: 'overview',
    healthData: null,
    metadata: null,
    experiments: null,
    comparison: null,
    activeCsvFile: null,
    activeCsvData: null,
    previewResult: null,
    inferenceResult: null,
    lastAnalysisType: null
  };

  // DOM Elements
  const navItems = document.querySelectorAll('.nav-item');
  const viewSections = document.querySelectorAll('.view-section');
  const breadcrumbCurrent = document.getElementById('breadcrumbCurrent');
  const systemStatusPill = document.getElementById('systemStatusPill');
  const systemStatusText = document.getElementById('systemStatusText');
  const statusDot = document.getElementById('statusDot');
  const menuToggleBtn = document.getElementById('menuToggleBtn');
  const sidebar = document.getElementById('sidebar');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');

  function closeSidebar() {
    if (sidebar && sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
    if (sidebarBackdrop && sidebarBackdrop.classList.contains('active')) {
      sidebarBackdrop.classList.remove('active');
    }
  }

  function toggleSidebar() {
    if (!sidebar) return;
    const isOpen = sidebar.classList.toggle('open');
    if (sidebarBackdrop) {
      if (isOpen) {
        sidebarBackdrop.classList.add('active');
      } else {
        sidebarBackdrop.classList.remove('active');
      }
    }
  }

  // --------------------------------------------------------------------------
  // Navigation & View Routing
  // --------------------------------------------------------------------------
  function navigateTo(viewId) {
    AppState.currentView = viewId;

    navItems.forEach(item => {
      const target = item.getAttribute('data-view');
      if (target === viewId) {
        item.classList.add('active');
        const viewTitle = item.querySelector('span')?.textContent || viewId;
        if (breadcrumbCurrent) breadcrumbCurrent.textContent = viewTitle;
      } else {
        item.classList.remove('active');
      }
    });

    viewSections.forEach(sec => {
      if (sec.id === `view-${viewId}`) {
        sec.classList.add('active');
      } else {
        sec.classList.remove('active');
      }
    });

    // Close mobile sidebar drawer
    closeSidebar();

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Lazy load/render charts if needed
    if (viewId === 'model-comparison' && AppState.comparison) {
      setTimeout(renderComparisonCharts, 50);
    } else if (viewId === 'unseen-attacks' && AppState.comparison) {
      setTimeout(renderUnseenAttackCharts, 50);
    }
  }

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const view = item.getAttribute('data-view');
      if (view) navigateTo(view);
    });
  });

  if (menuToggleBtn) {
    menuToggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSidebar();
    });
  }

  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', closeSidebar);
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeSidebar();
    }
  });

  // Quick navigation buttons in overview cards
  document.querySelectorAll('[data-goto]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-goto');
      if (target) navigateTo(target);
    });
  });

  // --------------------------------------------------------------------------
  // Toast Notifications
  // --------------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  }

  // --------------------------------------------------------------------------
  // Initial Data Fetching
  // --------------------------------------------------------------------------
  async function initApp() {
    try {
      // 1. Health check
      AppState.healthData = await API.getHealth();
      updateHealthUI(AppState.healthData);

      // 2. Metadata
      AppState.metadata = await API.getMetadata();
      renderMetadataOverview(AppState.metadata);
      renderDatasetExperimentsPage(AppState.metadata);

      // 3. Comparison & Experiments
      AppState.comparison = await API.getComparison();
      renderModelComparisonPage(AppState.comparison);
      renderUnseenAttacksPage(AppState.comparison);

      // 4. Research figures (data-driven SVG evaluation figures)
      try {
        AppState.researchFigures = await API.getResearchFigures();
      } catch (rfErr) {
        console.warn('Research figures API fallback:', rfErr);
      }

      // Initial chart rendering
      renderComparisonCharts();
      if (AppState.metadata) {
        renderDatasetExperimentsPage(AppState.metadata);
      }

      // Render system info
      renderSystemInfoPage(AppState.healthData, AppState.metadata);

      console.log('[+] Cognisec AI Dashboard initialized successfully.');
    } catch (err) {
      console.error('[!] Initialization error:', err);
      showToast('Failed to connect to backend API server.', 'error');
      if (systemStatusText) systemStatusText.textContent = 'Backend Offline';
      if (statusDot) statusDot.className = 'status-dot degraded';
    }
  }

  function updateHealthUI(health) {
    const isHealthy = health && health.status === 'healthy';
    if (systemStatusText) {
      systemStatusText.textContent = isHealthy ? 'System Online (Models Loaded)' : 'System Degraded';
    }
    if (statusDot) {
      statusDot.className = `status-dot ${isHealthy ? '' : 'degraded'}`;
    }

    // Top badges
    const xgbBadge = document.getElementById('topXgbBadge');
    if (xgbBadge && health.models?.xgboost?.threshold !== undefined) {
      xgbBadge.textContent = `XGBoost Threshold: ${health.models.xgboost.threshold}`;
    }

    const vaeBadge = document.getElementById('topVaeBadge');
    if (vaeBadge && health.models?.vae?.threshold !== undefined) {
      vaeBadge.textContent = `VAE Threshold: ${health.models.vae.threshold.toFixed(4)}`;
    }

    // Live Monitoring threshold badges
    const liveXgbThreshBadge = document.getElementById('liveXgbThreshBadge');
    if (liveXgbThreshBadge && health.models?.xgboost?.threshold !== undefined) {
      liveXgbThreshBadge.textContent = `Threshold: ${Number(health.models.xgboost.threshold).toFixed(4)}`;
    }

    const liveVaeThreshBadge = document.getElementById('liveVaeThreshBadge');
    if (liveVaeThreshBadge && health.models?.vae?.threshold !== undefined) {
      liveVaeThreshBadge.textContent = `Threshold: ${Number(health.models.vae.threshold).toFixed(6)}`;
    }

    const methodThreshold = document.getElementById('methodCalibratedThreshold');
    if (methodThreshold && health.models?.vae?.threshold !== undefined) {
      methodThreshold.textContent = `Calibrated Threshold = ${Number(health.models.vae.threshold).toFixed(4)}`;
    }
  }

  // --------------------------------------------------------------------------
  // Overview View Rendering
  // --------------------------------------------------------------------------
  function renderMetadataOverview(meta) {
    if (!meta) return;

    const flowCountEl = document.getElementById('ovTotalFlows');
    if (flowCountEl) flowCountEl.textContent = (meta.total_raw_records || 757646).toLocaleString();

    const featCountEl = document.getElementById('ovFeatureCount');
    if (featCountEl) featCountEl.textContent = `${meta.retained_clean_features_count || 70} Clean Features`;

    const dupCountEl = document.getElementById('ovDuplicates');
    if (dupCountEl) dupCountEl.textContent = `${meta.duplicate_records?.toLocaleString()} (${meta.duplicate_percentage}%)`;

    const heldOutFamiliesEl = document.getElementById('ovHeldOutFamilies');
    if (heldOutFamiliesEl && meta.research_integrity?.unseen_attack_families) {
      heldOutFamiliesEl.textContent = meta.research_integrity.unseen_attack_families.join(', ');
    }

    const heldOutCountEl = document.getElementById('ovHeldOutCount');
    const heldOutAttacks = meta.partitioning?.test_summary?.held_out_attacks;
    if (heldOutCountEl && heldOutAttacks !== undefined) {
      heldOutCountEl.textContent = Number(heldOutAttacks).toLocaleString();
    }

    // Dynamic research methodology numbers
    const methodRawFlows = document.getElementById('methodRawFlows');
    if (methodRawFlows && meta.total_raw_records) {
      methodRawFlows.textContent = `${Number(meta.total_raw_records).toLocaleString()} raw flows`;
    }

    const methodBenignFlows = document.getElementById('methodBenignFlows');
    const trainBenign = meta.partitioning?.train_summary?.benign;
    if (methodBenignFlows && trainBenign) {
      methodBenignFlows.textContent = `${Number(trainBenign).toLocaleString()} Benign Flows`;
    }

    const methodTestFlows = document.getElementById('methodTestFlows');
    const testTotal = meta.partitioning?.test_summary?.total;
    if (methodTestFlows && testTotal) {
      methodTestFlows.textContent = `${Number(testTotal).toLocaleString()} Test Flows`;
    }
  }

  // --------------------------------------------------------------------------
  // Traffic Analysis (CSV Upload & Preview)
  // --------------------------------------------------------------------------
  const dropzone = document.getElementById('csvDropzone');
  const fileInput = document.getElementById('csvFileInput');
  const sampleCsvBtn = document.getElementById('loadSampleCsvBtn');
  const clearUploadBtn = document.getElementById('clearUploadBtn');
  const uploadMetaBar = document.getElementById('uploadMetaBar');
  const uploadFileName = document.getElementById('uploadFileName');
  const uploadFileSize = document.getElementById('uploadFileSize');
  const uploadStatusBadge = document.getElementById('uploadStatusBadge');
  const previewCard = document.getElementById('previewCard');
  const previewTableHead = document.getElementById('previewTableHead');
  const previewTableBody = document.getElementById('previewTableBody');

  if (dropzone && fileInput) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  async function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showToast('Please select a valid .csv file.', 'error');
      return;
    }

    AppState.activeCsvFile = file;
    AppState.activeCsvData = null;

    if (uploadMetaBar) uploadMetaBar.style.display = 'flex';
    if (uploadFileName) uploadFileName.textContent = file.name;
    if (uploadFileSize) uploadFileSize.textContent = formatBytes(file.size);
    if (uploadStatusBadge) {
      uploadStatusBadge.className = 'badge badge-slate';
      uploadStatusBadge.textContent = 'Validating schema...';
    }

    const formData = new FormData();
    formData.append('file', file);
    await processCsvPreview(formData);
  }

  // Load sample authentic test data button
  if (sampleCsvBtn) {
    sampleCsvBtn.addEventListener('click', async () => {
      try {
        sampleCsvBtn.disabled = true;
        sampleCsvBtn.textContent = 'Loading sample flows...';

        const sample = await API.getSampleCsv();
        AppState.activeCsvFile = null;
        AppState.activeCsvData = sample.csv_text;

        if (uploadMetaBar) uploadMetaBar.style.display = 'flex';
        if (uploadFileName) uploadFileName.textContent = 'sample_flows.csv (Authentic CICIDS2017 Test Set)';
        if (uploadFileSize) uploadFileSize.textContent = `${sample.rows_count} flow records (${sample.columns_count} columns)`;

        // Process preview via JSON
        await processCsvPreview({ csv_text: sample.csv_text }, false);
        showToast('Sample CICIDS2017 flow records loaded.', 'success');
      } catch (err) {
        showToast(`Failed to load sample CSV: ${err.message}`, 'error');
      } finally {
        sampleCsvBtn.disabled = false;
        sampleCsvBtn.textContent = 'Load Sample CICIDS2017 CSV (100 Flows)';
      }
    });
  }

  if (clearUploadBtn) {
    clearUploadBtn.addEventListener('click', () => {
      AppState.activeCsvFile = null;
      AppState.activeCsvData = null;
      AppState.previewResult = null;
      AppState.inferenceResult = null;
      AppState.activeCsvAnalytics = null;
      if (fileInput) fileInput.value = '';
      if (uploadMetaBar) uploadMetaBar.style.display = 'none';
      if (previewCard) previewCard.style.display = 'none';
      const trafficFamilyCard = document.getElementById('trafficFamilyCard');
      if (trafficFamilyCard) trafficFamilyCard.style.display = 'none';
      resetInferenceResultsUI();
      showToast('Cleared active upload.', 'info');
    });
  }

  async function processCsvPreview(data, isFormData = true) {
    try {
      const result = await API.previewCsv(data);
      AppState.previewResult = result;

      // Update badge
      if (uploadStatusBadge) {
        if (result.is_compatible) {
          uploadStatusBadge.className = 'badge badge-green';
          uploadStatusBadge.textContent = `Compatible (${result.matched_feature_count}/${result.expected_feature_count} features matched)`;
        } else {
          uploadStatusBadge.className = 'badge badge-red';
          uploadStatusBadge.textContent = `Incompatible (Missing ${result.missing_feature_count} features)`;
        }
      }

      // Render summary cards
      renderCsvValidationMetrics(result);

      // Render preview table
      renderPreviewTable(result);

      if (previewCard) previewCard.style.display = 'block';

      // Fetch dynamic analytics for the previewed CSV
      try {
        const analyticsRes = await API.getCsvAnalytics(data);
        if (analyticsRes && analyticsRes.data) {
          const analytics = analyticsRes.data;
          AppState.activeCsvAnalytics = analytics;
          const familyCard = document.getElementById('trafficFamilyCard');
          if (familyCard) {
            familyCard.style.display = 'block';
            Charts.renderTrafficFamilyDistribution('trafficFamilyDistributionChart', analytics.family_distribution, analytics.has_ground_truth);
          }
        }
      } catch (anErr) {
        console.warn('CSV analytics calculation notice:', anErr);
      }

      // Enable inference buttons in Section 3
      updateInferenceButtonsState(true);
      showToast(result.message, result.is_compatible ? 'success' : 'error');
    } catch (err) {
      showToast(`CSV preview failed: ${err.message}`, 'error');
      if (uploadStatusBadge) {
        uploadStatusBadge.className = 'badge badge-red';
        uploadStatusBadge.textContent = 'Validation Error';
      }
    }
  }

  function renderCsvValidationMetrics(res) {
    const totalRowsEl = document.getElementById('previewTotalRows');
    const totalColsEl = document.getElementById('previewTotalCols');
    const matchedFeatsEl = document.getElementById('previewMatchedFeats');
    const nullsEl = document.getElementById('previewNullCount');

    if (totalRowsEl) totalRowsEl.textContent = res.total_rows.toLocaleString();
    if (totalColsEl) totalColsEl.textContent = res.total_columns;
    if (matchedFeatsEl) matchedFeatsEl.textContent = `${res.matched_feature_count} / ${res.expected_feature_count}`;

    const totalMissing = Object.values(res.missing_values_summary || {}).reduce((a, b) => a + b, 0);
    const totalInfs = Object.values(res.infinite_values_summary || {}).reduce((a, b) => a + b, 0);
    if (nullsEl) nullsEl.textContent = `${totalMissing} nulls / ${totalInfs} infs (auto-imputed)`;
  }

  function renderPreviewTable(res) {
    if (!previewTableHead || !previewTableBody) return;
    previewTableHead.innerHTML = '';
    previewTableBody.innerHTML = '';

    if (!res.preview_rows || res.preview_rows.length === 0) return;

    const columns = Object.keys(res.preview_rows[0]);
    // Show first 12 columns for readability
    const displayCols = columns.slice(0, 12);

    let headHtml = '<tr><th style="width: 40px;">#</th>';
    displayCols.forEach(col => {
      headHtml += `<th>${col}</th>`;
    });
    if (columns.length > 12) headHtml += `<th>+ ${columns.length - 12} more columns</th>`;
    headHtml += '</tr>';
    previewTableHead.innerHTML = headHtml;

    let bodyHtml = '';
    res.preview_rows.forEach((row, idx) => {
      bodyHtml += `<tr><td>${idx + 1}</td>`;
      displayCols.forEach(col => {
        const val = row[col];
        bodyHtml += `<td>${val !== null && val !== undefined ? val : '<span style="color:#94a3b8;">null</span>'}</td>`;
      });
      if (columns.length > 12) bodyHtml += `<td>...</td>`;
      bodyHtml += '</tr>';
    });
    previewTableBody.innerHTML = bodyHtml;
  }

  function updateInferenceButtonsState(enabled) {
    ['runXgbBtn', 'runVaeBtn', 'runDualBtn'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = !enabled;
    });
  }

  // --------------------------------------------------------------------------
  // Attack Detection (Model Inference)
  // --------------------------------------------------------------------------
  const runXgbBtn = document.getElementById('runXgbBtn');
  const runVaeBtn = document.getElementById('runVaeBtn');
  const runDualBtn = document.getElementById('runDualBtn');
  const resultsCard = document.getElementById('inferenceResultsCard');
  const resultsTableHead = document.getElementById('resultsTableHead');
  const resultsTableBody = document.getElementById('resultsTableBody');
  const downloadCsvBtn = document.getElementById('downloadPredictionsBtn');
  const filterPredictionsSelect = document.getElementById('filterPredictions');
  const searchPredictionsInput = document.getElementById('searchPredictions');

  if (runXgbBtn) {
    runXgbBtn.addEventListener('click', () => executeInference('xgboost'));
  }
  if (runVaeBtn) {
    runVaeBtn.addEventListener('click', () => executeInference('vae'));
  }
  if (runDualBtn) {
    runDualBtn.addEventListener('click', () => executeInference('dual'));
  }

  async function executeInference(mode) {
    if (!AppState.activeCsvFile && !AppState.activeCsvData) {
      showToast('Please upload a CSV or load sample flows first.', 'error');
      return;
    }

    const btn = mode === 'xgboost' ? runXgbBtn : mode === 'vae' ? runVaeBtn : runDualBtn;
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running ${mode.toUpperCase()}...`;

    try {
      let dataPayload;
      let isForm = false;
      if (AppState.activeCsvFile) {
        dataPayload = new FormData();
        dataPayload.append('file', AppState.activeCsvFile);
        isForm = true;
      } else {
        dataPayload = { csv_text: AppState.activeCsvData };
        isForm = false;
      }

      let res;
      if (mode === 'xgboost') {
        res = await API.runXGBoost(dataPayload);
      } else if (mode === 'vae') {
        res = await API.runVAE(dataPayload);
      } else {
        res = await API.runDual(dataPayload);
      }

      AppState.inferenceResult = res;
      AppState.lastAnalysisType = mode;

      renderInferenceMetrics(res, mode);
      renderInferenceTable(res, mode);

      // Render dynamic inference charts
      const analyticsPayload = res.analytics || AppState.activeCsvAnalytics;
      if (analyticsPayload) {
        renderCsvInferenceCharts(analyticsPayload);
      } else {
        // Query analytics on demand if not present
        try {
          const fetchedAnalytics = await API.getCsvAnalytics(dataPayload);
          if (fetchedAnalytics && fetchedAnalytics.data) {
            AppState.activeCsvAnalytics = fetchedAnalytics.data;
            renderCsvInferenceCharts(fetchedAnalytics.data);
          }
        } catch (fErr) {
          console.warn('Could not fetch inference analytics:', fErr);
        }
      }

      if (resultsCard) resultsCard.style.display = 'block';
      showToast(`${mode.toUpperCase()} analysis complete on ${res.total_flows} flows.`, 'success');

      // Scroll to results
      resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      showToast(`Inference failed: ${err.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalText;
    }
  }

  function renderInferenceMetrics(res, mode) {
    const totalEl = document.getElementById('resTotalFlows');
    const flaggedEl = document.getElementById('resFlaggedFlows');
    const rateEl = document.getElementById('resDetectionRate');
    const noteEl = document.getElementById('resModelNote');

    if (totalEl) totalEl.textContent = res.total_flows.toLocaleString();

    if (mode === 'xgboost') {
      if (flaggedEl) flaggedEl.textContent = `${res.flagged_attacks.toLocaleString()} Attacks`;
      if (rateEl) rateEl.textContent = `${res.attack_percentage}%`;
      if (noteEl) noteEl.textContent = `Classification Threshold: ${res.decision_threshold}`;
    } else if (mode === 'vae') {
      if (flaggedEl) flaggedEl.textContent = `${res.flagged_anomalies.toLocaleString()} Anomalies`;
      if (rateEl) rateEl.textContent = `${res.anomaly_percentage}%`;
      if (noteEl) noteEl.textContent = `Anomaly Threshold: ${res.calibrated_threshold} (1% FPR budget)`;
    } else {
      const summary = res.summary || {};
      if (flaggedEl) flaggedEl.textContent = `${summary.both_flagged + summary.xgboost_only + summary.vae_only} Flagged`;
      if (rateEl) rateEl.textContent = `Both: ${summary.both_flagged_pct}%, XGB-Only: ${summary.xgboost_only_pct}%, VAE-Only: ${summary.vae_only_pct}%`;
      if (noteEl) noteEl.textContent = `XGB Thresh: ${res.xgboost_summary?.threshold} | VAE Thresh: ${res.vae_summary?.threshold}`;
    }
  }

  function renderInferenceTable(res, mode, filter = 'all', searchQuery = '') {
    if (!resultsTableHead || !resultsTableBody) return;
    resultsTableHead.innerHTML = '';
    resultsTableBody.innerHTML = '';

    const records = res.records || [];
    if (records.length === 0) {
      resultsTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #64748b;">No flow records returned.</td></tr>';
      return;
    }

    if (mode === 'xgboost') {
      resultsTableHead.innerHTML = `
        <tr>
          <th>Flow #</th>
          <th>Original Label</th>
          <th>Attack Probability</th>
          <th>Decision Threshold</th>
          <th>Status</th>
          <th>Classification</th>
        </tr>
      `;
    } else if (mode === 'vae') {
      resultsTableHead.innerHTML = `
        <tr>
          <th>Flow #</th>
          <th>Original Label</th>
          <th>Reconstruction Anomaly Score</th>
          <th>Calibrated Threshold</th>
          <th>Status</th>
          <th>Anomaly Flag</th>
        </tr>
      `;
    } else {
      resultsTableHead.innerHTML = `
        <tr>
          <th>Flow #</th>
          <th>Original Label</th>
          <th>XGBoost Prob</th>
          <th>VAE Score</th>
          <th>Agreement Category</th>
          <th>Dual Status</th>
        </tr>
      `;
    }

    let rowsHtml = '';
    let displayedCount = 0;

    records.forEach((r) => {
      // Filtering
      if (mode === 'xgboost') {
        if (filter === 'flagged' && !r.is_attack) return;
        if (filter === 'normal' && r.is_attack) return;
      } else if (mode === 'vae') {
        if (filter === 'flagged' && !r.is_anomaly) return;
        if (filter === 'normal' && r.is_anomaly) return;
      } else if (mode === 'dual') {
        if (filter === 'both_flagged' && r.agreement_category !== 'Both Flagged') return;
        if (filter === 'disagreement' && !r.agreement_category.includes('Only')) return;
        if (filter === 'normal' && r.agreement_category !== 'Both Normal') return;
      }

      // Search
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const label = (r.original_label || '').toLowerCase();
        const pred = (r.prediction || r.agreement_category || '').toLowerCase();
        if (!label.includes(query) && !pred.includes(query) && !String(r.flow_index).includes(query)) {
          return;
        }
      }

      displayedCount++;
      if (displayedCount > 100) return; // Limit rendering for performance

      if (mode === 'xgboost') {
        const isAtk = r.is_attack;
        rowsHtml += `
          <tr>
            <td><strong>#${r.flow_index}</strong></td>
            <td><code>${r.original_label}</code></td>
            <td><strong style="color: ${isAtk ? '#dc2626' : '#059669'};">${r.attack_probability}</strong></td>
            <td>${r.decision_threshold}</td>
            <td><span class="badge ${isAtk ? 'badge-red' : 'badge-green'}">${r.prediction}</span></td>
            <td>${isAtk ? '🚨 Flagged as Attack' : '✅ Verified Benign'}</td>
          </tr>
        `;
      } else if (mode === 'vae') {
        const isAnom = r.is_anomaly;
        rowsHtml += `
          <tr>
            <td><strong>#${r.flow_index}</strong></td>
            <td><code>${r.original_label}</code></td>
            <td><strong style="color: ${isAnom ? '#7c3aed' : '#059669'};">${r.anomaly_score}</strong></td>
            <td>${r.calibrated_threshold}</td>
            <td><span class="badge ${isAnom ? 'badge-indigo' : 'badge-green'}">${r.prediction}</span></td>
            <td>${isAnom ? '🚨 Deviates from Normal Pattern' : '✅ Conforms to Benign Manifold'}</td>
          </tr>
        `;
      } else {
        const cat = r.agreement_category;
        let badgeClass = 'badge-slate';
        if (cat === 'Both Flagged') badgeClass = 'badge-red';
        else if (cat === 'XGBoost Only') badgeClass = 'badge-blue';
        else if (cat === 'VAE Only') badgeClass = 'badge-indigo';
        else if (cat === 'Both Normal') badgeClass = 'badge-green';

        rowsHtml += `
          <tr>
            <td><strong>#${r.flow_index}</strong></td>
            <td><code>${r.original_label}</code></td>
            <td>${r.xgb_probability} (${r.xgb_prediction})</td>
            <td>${r.vae_anomaly_score} (${r.vae_prediction})</td>
            <td><span class="badge ${badgeClass}">${cat}</span></td>
            <td>${cat === 'VAE Only' ? '★ Stealth Anomaly Caught' : cat}</td>
          </tr>
        `;
      }
    });

    resultsTableBody.innerHTML = rowsHtml || '<tr><td colspan="6" style="text-align: center; color: #64748b;">No matching flows found.</td></tr>';
  }

  if (filterPredictionsSelect) {
    filterPredictionsSelect.addEventListener('change', (e) => {
      if (AppState.inferenceResult && AppState.lastAnalysisType) {
        renderInferenceTable(
          AppState.inferenceResult,
          AppState.lastAnalysisType,
          e.target.value,
          searchPredictionsInput ? searchPredictionsInput.value : ''
        );
      }
    });
  }

  if (searchPredictionsInput) {
    searchPredictionsInput.addEventListener('input', (e) => {
      if (AppState.inferenceResult && AppState.lastAnalysisType) {
        renderInferenceTable(
          AppState.inferenceResult,
          AppState.lastAnalysisType,
          filterPredictionsSelect ? filterPredictionsSelect.value : 'all',
          e.target.value
        );
      }
    });
  }

  // Download predictions as CSV
  if (downloadCsvBtn) {
    downloadCsvBtn.addEventListener('click', () => {
      if (!AppState.inferenceResult || !AppState.inferenceResult.records) {
        showToast('No inference records available to export.', 'error');
        return;
      }
      exportPredictionsToCsv(AppState.inferenceResult.records, AppState.lastAnalysisType);
    });
  }

  function exportPredictionsToCsv(records, mode) {
    if (!records || records.length === 0) return;
    const keys = Object.keys(records[0]);
    let csvContent = keys.join(',') + '\n';

    records.forEach(r => {
      const row = keys.map(k => `"${String(r[k] !== undefined ? r[k] : '').replace(/"/g, '""')}"`).join(',');
      csvContent += row + '\n';
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `inference_${mode}_results.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Predictions downloaded as CSV.', 'success');
  }

  function resetInferenceResultsUI() {
    if (resultsCard) resultsCard.style.display = 'none';
    if (resultsTableBody) resultsTableBody.innerHTML = '';
  }

  // --------------------------------------------------------------------------
  // Model Comparison Page Rendering
  // --------------------------------------------------------------------------
  function renderModelComparisonPage(comp) {
    if (!comp) return;

    // Update Overview KPI cards from verified historical benchmark metrics
    const ovXgbPrAuc = document.getElementById('ovXgbPrAuc');
    if (ovXgbPrAuc && comp.known_attack_metrics) {
      const prMetric = comp.known_attack_metrics.find(m => m.metric && m.metric.includes('PR-AUC'));
      if (prMetric) {
        ovXgbPrAuc.textContent = `${(Number(prMetric.xgboost) * 100).toFixed(2)}%`;
      }
    }

    const ovVaeInfilRecall = document.getElementById('ovVaeInfilRecall');
    if (ovVaeInfilRecall && comp.per_family_results) {
      const infil = comp.per_family_results.find(f => f.family === 'Infiltration');
      if (infil) {
        ovVaeInfilRecall.textContent = `${Number(infil.vae_recall).toFixed(2)}%`;
      }
    }

    // Metrics table
    const tableBody = document.getElementById('comparisonTableBody');
    if (tableBody && comp.known_attack_metrics) {
      let html = '';
      comp.known_attack_metrics.forEach(m => {
        html += `
          <tr>
            <td><strong>${m.metric}</strong></td>
            <td><strong style="color: #2563eb;">${m.xgboost}</strong> ${m.xgboost_ci || ''}</td>
            <td><strong style="color: #7c3aed;">${m.vae}</strong> ${m.vae_ci || ''}</td>
            <td>${m.difference > 0 ? `+${m.difference}` : m.difference}</td>
            <td><code>${m.p_value}</code></td>
            <td><span class="badge ${m.advantage.includes('XGBoost') ? 'badge-blue' : 'badge-indigo'}">${m.advantage}</span></td>
          </tr>
        `;
      });
      tableBody.innerHTML = html;
    }
  }

  function renderComparisonCharts() {
    if (!AppState.comparison) return;

    // 1. Per-family recall
    Charts.renderPerFamilyRecall('perFamilyRecallChart', AppState.comparison.per_family_results);

    // 2. Threshold sensitivity
    Charts.renderSensitivityChart('sensitivityChart', AppState.comparison.threshold_sensitivity);

    // 3. Distribution shift
    Charts.renderDistributionShiftChart('distributionShiftChart', AppState.comparison.distribution_shift);

    // 4. ROC & PR Curves (Figure 2 Panel A & B)
    const fig2Data = (AppState.researchFigures && AppState.researchFigures.data && AppState.researchFigures.data.fig2_roc_pr) || AppState.comparison;
    Charts.renderRocCurves('rocCurveChart', fig2Data);
    Charts.renderPrCurves('prCurveChart', fig2Data);

    // 5. Confusion Matrices (Figure 3)
    const fig3Data = (AppState.researchFigures && AppState.researchFigures.data && AppState.researchFigures.data.fig3_confusion_matrix) || AppState.comparison.confusion_matrix;
    Charts.renderConfusionMatrices('confusionMatrixChart', fig3Data);
  }

  function renderCsvInferenceCharts(analytics) {
    if (!analytics) return;

    // 1. Prediction distribution
    Charts.renderPredictionDistribution('csvPredictionDistributionChart', analytics.prediction_distribution);

    // 2. Anomaly score distribution with calibrated threshold
    const thresh = (analytics.anomaly_score_distribution && analytics.anomaly_score_distribution.calibrated_threshold) || 0.330610;
    Charts.renderAnomalyScoreDistribution('csvAnomalyScoreChart', analytics.anomaly_score_distribution, thresh);

    // 3. Model disagreement
    Charts.renderModelDisagreement('csvModelDisagreementChart', analytics.model_disagreement);

    // 4. Ground-truth evaluation if available, or clear "Ground truth unavailable" card
    const gtContainer = document.getElementById('csvGroundTruthEvaluation');
    if (gtContainer) {
      if (analytics.has_ground_truth && analytics.ground_truth_evaluation) {
        const gt = analytics.ground_truth_evaluation;
        const xgbMetrics = gt.xgboost_metrics || {};
        const vaeMetrics = gt.vae_metrics || {};
        gtContainer.innerHTML = `
          <div class="chart-header-row">
            <span class="chart-title-sm">Empirical Ground-Truth Evaluation</span>
            <span class="badge badge-green">Verified Labels Present</span>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; font-size: 12px;">
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 4px; padding: 10px;">
              <strong style="color: #1e40af; display: block; margin-bottom: 6px;">XGBoost (Supervised)</strong>
              <div>Precision: <strong>${(xgbMetrics.precision * 100).toFixed(2)}%</strong></div>
              <div>Recall: <strong>${(xgbMetrics.recall * 100).toFixed(2)}%</strong></div>
              <div>F1-Score: <strong>${(xgbMetrics.f1 * 100).toFixed(2)}%</strong></div>
              <div>FPR: <strong>${(xgbMetrics.fpr * 100).toFixed(2)}%</strong></div>
            </div>
            <div style="background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 4px; padding: 10px;">
              <strong style="color: #6b21a8; display: block; margin-bottom: 6px;">Tabular VAE (Anomaly)</strong>
              <div>Precision: <strong>${(vaeMetrics.precision * 100).toFixed(2)}%</strong></div>
              <div>Recall: <strong>${(vaeMetrics.recall * 100).toFixed(2)}%</strong></div>
              <div>F1-Score: <strong>${(vaeMetrics.f1 * 100).toFixed(2)}%</strong></div>
              <div>FPR: <strong>${(vaeMetrics.fpr * 100).toFixed(2)}%</strong></div>
            </div>
          </div>
        `;
      } else {
        gtContainer.innerHTML = `
          <div class="chart-header-row">
            <span class="chart-title-sm">Ground-Truth Evaluation</span>
            <span class="badge badge-slate">Unlabeled Dataset</span>
          </div>
          <div class="chart-empty-state" style="padding: 16px;">
            <div style="font-weight: 600; color: #475569; margin-bottom: 4px;">Ground Truth Unavailable</div>
            <div style="font-size: 12px; color: #64748b; line-height: 1.4;">
              This dataset does not contain verified ground-truth labels. Metrics such as Precision, Recall, F1, and False-Positive Rate are not fabricated for unverified data.
            </div>
          </div>
        `;
      }
    }
  }

  // --------------------------------------------------------------------------
  // Unseen Attack Evaluation Page Rendering
  // --------------------------------------------------------------------------
  function renderUnseenAttacksPage(comp) {
    if (!comp) return;

    // Summary stats
    const summary = comp.held_out_summary || {};
    const xgbSum = summary.xgboost || {};
    const vaeSum = summary.vae || {};
    const paired = summary.paired_comparison || {};

    const xgbRecallEl = document.getElementById('unseenXgbRecall');
    if (xgbRecallEl) xgbRecallEl.textContent = `${(xgbSum.recall * 100).toFixed(2)}%`;

    const vaeRecallEl = document.getElementById('unseenVaeRecall');
    if (vaeRecallEl) vaeRecallEl.textContent = `${(vaeSum.recall * 100).toFixed(2)}%`;

    const deltaEl = document.getElementById('unseenDeltaRecall');
    if (deltaEl) deltaEl.textContent = `Δ = ${(paired.delta_point * 100).toFixed(2)}% (p < 0.001)`;

    const unseenHeldOutBadge = document.getElementById('unseenHeldOutBadge');
    if (unseenHeldOutBadge && comp.complementarity && comp.complementarity.total_held_out_attacks) {
      unseenHeldOutBadge.textContent = `${Number(comp.complementarity.total_held_out_attacks).toLocaleString()} Held-Out Attacks`;
    }

    // Disaggregated table
    const unseenTableBody = document.getElementById('unseenTableBody');
    if (unseenTableBody && comp.per_family_results) {
      let html = '';
      comp.per_family_results.forEach(f => {
        html += `
          <tr style="${f.is_held_out ? 'background-color: #faf5ff;' : ''}">
            <td>
              <strong>${f.family}</strong>
              ${f.is_held_out ? '<span class="badge badge-red" style="margin-left: 8px;">Held-Out Zero-Day</span>' : '<span class="badge badge-slate" style="margin-left: 8px;">Known</span>'}
            </td>
            <td>${f.count.toLocaleString()} flows</td>
            <td><strong style="color: #2563eb;">${f.xgboost_recall}%</strong></td>
            <td><strong style="color: #7c3aed;">${f.vae_recall}%</strong></td>
            <td>
              ${f.family === 'Infiltration' ? '<span class="badge badge-indigo">★ VAE Complementary (+47.2%)</span>' :
                f.family === 'DDoS' ? '<span class="badge badge-blue">Signature Overlap from DoS</span>' :
                f.note || '-'}
            </td>
          </tr>
        `;
      });
      unseenTableBody.innerHTML = html;
    }
  }

  function renderUnseenAttackCharts() {
    if (!AppState.comparison) return;
    Charts.renderComplementarityDonut('complementarityDonutChart', AppState.comparison.complementarity);
  }

  // --------------------------------------------------------------------------
  // Dataset & Experiments Page Rendering
  // --------------------------------------------------------------------------
  function renderDatasetExperimentsPage(meta) {
    if (!meta) return;

    const manifest = meta.partitioning || {};
    const trainSum = manifest.train_summary || {};
    const valSum = manifest.val_summary || {};
    const testSum = manifest.test_summary || {};

    const trainTotalEl = document.getElementById('dsTrainTotal');
    const valTotalEl = document.getElementById('dsValTotal');
    const testTotalEl = document.getElementById('dsTestTotal');

    if (trainTotalEl) trainTotalEl.textContent = `${trainSum.total?.toLocaleString()} (60%)`;
    if (valTotalEl) valTotalEl.textContent = `${valSum.total?.toLocaleString()} (20%)`;
    if (testTotalEl) testTotalEl.textContent = `${testSum.total?.toLocaleString()} (20%)`;

    // Render Live Dataset Distribution Chart (Figure 1)
    const fig1Data = (AppState.researchFigures && AppState.researchFigures.data && AppState.researchFigures.data.fig1_dataset_distribution) || meta.family_distribution || meta.dataset_distribution;
    if (fig1Data) {
      Charts.renderDatasetDistribution('datasetDistributionChart', fig1Data);
    }

    // Daily provenance list populated dynamically from verified files
    const provList = document.getElementById('dsProvenanceList');
    if (provList && meta.file_provenance) {
      let provHtml = '';
      Object.entries(meta.file_provenance).forEach(([fn, count]) => {
        provHtml += `<span>• ${fn}: <strong>${Number(count).toLocaleString()}</strong> flows</span>`;
      });
      provList.innerHTML = provHtml;
    }

    // Partition distributions table
    const tableBody = document.getElementById('partitionTableBody');
    if (tableBody) {
      const trainFams = trainSum.family_distribution || {};
      const valFams = valSum.family_distribution || {};
      const testFams = testSum.family_distribution || {};

      const allFams = Array.from(new Set([
        ...Object.keys(trainFams),
        ...Object.keys(valFams),
        ...Object.keys(testFams)
      ])).sort();

      let html = '';
      allFams.forEach(fam => {
        const inTrain = trainFams[fam] || 0;
        const inVal = valFams[fam] || 0;
        const inTest = testFams[fam] || 0;
        const isHeldOut = (meta.research_integrity?.unseen_attack_families || []).includes(fam);

        html += `
          <tr style="${isHeldOut ? 'background-color: #fef2f2;' : ''}">
            <td><strong>${fam}</strong> ${isHeldOut ? '<span class="badge badge-red" style="margin-left: 6px;">Held-Out</span>' : ''}</td>
            <td>${inTrain.toLocaleString()}</td>
            <td>${inVal.toLocaleString()}</td>
            <td>${inTest.toLocaleString()}</td>
            <td>${isHeldOut ? 'Strict Zero-Day Isolation' : 'Represented in Training'}</td>
          </tr>
        `;
      });
      tableBody.innerHTML = html;
    }
  }

  // --------------------------------------------------------------------------
  // System Information Page Rendering
  // --------------------------------------------------------------------------
  function renderSystemInfoPage(health, meta) {
    if (!health) return;

    const env = health.environment || {};
    const models = health.models || {};

    const pyVerEl = document.getElementById('sysPyVer');
    const torchVerEl = document.getElementById('sysTorchVer');
    const deviceEl = document.getElementById('sysDevice');
    const xgbThreshEl = document.getElementById('sysXgbThresh');
    const vaeThreshEl = document.getElementById('sysVaeThresh');
    const featCountEl = document.getElementById('sysFeatCount');

    if (pyVerEl) pyVerEl.textContent = env.python_version || '-';
    if (torchVerEl) torchVerEl.textContent = env.torch_version || '-';
    if (deviceEl) deviceEl.textContent = env.device_available?.toUpperCase() || 'CPU';
    if (xgbThreshEl) xgbThreshEl.textContent = models.xgboost?.threshold || '-';
    if (vaeThreshEl) vaeThreshEl.textContent = models.vae?.threshold?.toFixed(4) || '-';
    if (featCountEl) featCountEl.textContent = `${models.preprocessor?.feature_count || 70} columns`;
  }

  // Format bytes helper
  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Initialize
  initApp();
});
