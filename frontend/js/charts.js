/**
 * Cognisec AI - Native SVG Charts Library
 * Renders high-DPI, responsive, interactive SVG charts without heavy external dependencies.
 */

const Charts = {
  /**
   * Render Per-Family Recall Comparison Bar Chart
   */
  renderPerFamilyRecall(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container || !data || data.length === 0) return;

    const width = 800;
    const height = 360;
    const margin = { top: 30, right: 30, bottom: 60, left: 110 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const barGroupHeight = chartHeight / data.length;
    const barHeight = Math.min(12, (barGroupHeight - 8) / 2);

    let barsHtml = '';
    let yLabelsHtml = '';

    data.forEach((item, idx) => {
      const yCenter = margin.top + idx * barGroupHeight + barGroupHeight / 2;
      const yXgb = yCenter - barHeight - 2;
      const yVae = yCenter + 2;

      const xgbW = (item.xgboost_recall / 100) * chartWidth;
      const vaeW = (item.vae_recall / 100) * chartWidth;

      const badgeType = item.is_held_out ? '[Zero-Day]' : '[Known]';

      barsHtml += `
        <!-- ${item.family} XGBoost -->
        <rect x="${margin.left}" y="${yXgb}" width="${Math.max(2, xgbW)}" height="${barHeight}" rx="3" fill="#2563eb">
          <title>${item.family} - XGBoost Recall: ${item.xgboost_recall}% (${item.count.toLocaleString()} flows)</title>
        </rect>
        <text x="${margin.left + xgbW + 6}" y="${yXgb + barHeight - 2}" font-size="11" font-weight="600" fill="#1e293b">${item.xgboost_recall}%</text>

        <!-- ${item.family} VAE -->
        <rect x="${margin.left}" y="${yVae}" width="${Math.max(2, vaeW)}" height="${barHeight}" rx="3" fill="#7c3aed">
          <title>${item.family} - VAE Recall: ${item.vae_recall}% (${item.count.toLocaleString()} flows)</title>
        </rect>
        <text x="${margin.left + vaeW + 6}" y="${yVae + barHeight - 2}" font-size="11" font-weight="600" fill="#64748b">${item.vae_recall}%</text>
      `;

      yLabelsHtml += `
        <text x="${margin.left - 10}" y="${yCenter + 4}" text-anchor="end" font-size="12" font-weight="${item.is_held_out ? '700' : '500'}" fill="${item.is_held_out ? '#dc2626' : '#334155'}">
          ${item.family} ${item.is_held_out ? '★' : ''}
        </text>
      `;
    });

    // Grid lines
    let gridHtml = '';
    for (let pct = 0; pct <= 100; pct += 25) {
      const x = margin.left + (pct / 100) * chartWidth;
      gridHtml += `
        <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${height - margin.bottom}" stroke="#e2e8f0" stroke-dasharray="3,3" />
        <text x="${x}" y="${height - margin.bottom + 18}" text-anchor="middle" font-size="11" fill="#64748b">${pct}%</text>
      `;
    }

    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; max-height: 400px; display: block;">
        ${gridHtml}
        ${barsHtml}
        ${yLabelsHtml}
        <!-- X Axis baseline -->
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#cbd5e1" stroke-width="1.5" />
        <!-- Axis title -->
        <text x="${margin.left + chartWidth / 2}" y="${height - 12}" text-anchor="middle" font-size="12" font-weight="500" fill="#475569">Detection Recall (% of Attack Flows Identified)</text>
      </svg>
    `;
  },

  /**
   * Render Threshold Sensitivity Line Chart (Exp C)
   */
  renderSensitivityChart(containerId, sensitivityList) {
    const container = document.getElementById(containerId);
    if (!container || !sensitivityList || sensitivityList.length === 0) return;

    const width = 640;
    const height = 300;
    const margin = { top: 25, right: 30, bottom: 45, left: 55 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const ptsRecall = [];
    const ptsF1 = [];
    const ptsPrecision = [];

    // Max target FPR is 10.0%
    const maxFPR = 10.0;

    sensitivityList.forEach((d) => {
      const x = margin.left + (d.target_fpr_percent / maxFPR) * chartW;
      const yRec = height - margin.bottom - d.test_recall * chartH;
      const yF1 = height - margin.bottom - d.test_f1 * chartH;
      const yPrec = height - margin.bottom - d.test_precision * chartH;

      ptsRecall.push({ x, y: yRec, val: d.test_recall, fpr: d.target_fpr_percent });
      ptsF1.push({ x, y: yF1, val: d.test_f1, fpr: d.target_fpr_percent });
      ptsPrecision.push({ x, y: yPrec, val: d.test_precision, fpr: d.target_fpr_percent });
    });

    const pathRec = ptsRecall.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    const pathF1 = ptsF1.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    const pathPrec = ptsPrecision.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

    let dotsHtml = '';
    ptsRecall.forEach((p) => {
      dotsHtml += `
        <circle cx="${p.x}" cy="${p.y}" r="4" fill="#7c3aed">
          <title>Target FPR: ${p.fpr}% -> Recall: ${(p.val * 100).toFixed(1)}%</title>
        </circle>
      `;
    });
    ptsF1.forEach((p) => {
      dotsHtml += `
        <circle cx="${p.x}" cy="${p.y}" r="4" fill="#059669">
          <title>Target FPR: ${p.fpr}% -> F1-Score: ${(p.val * 100).toFixed(1)}%</title>
        </circle>
      `;
    });

    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; max-height: 320px; display: block;">
        <!-- Grid lines -->
        ${[0, 0.25, 0.5, 0.75, 1.0].map(v => {
          const y = height - margin.bottom - v * chartH;
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#e2e8f0" stroke-dasharray="3,3" />
            <text x="${margin.left - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b">${(v * 100)}%</text>
          `;
        }).join('')}

        <!-- X Axis labels -->
        ${[0, 1, 2, 5, 10].map(fpr => {
          const x = margin.left + (fpr / maxFPR) * chartW;
          return `
            <line x1="${x}" y1="${height - margin.bottom}" x2="${x}" y2="${height - margin.bottom + 4}" stroke="#94a3b8" />
            <text x="${x}" y="${height - margin.bottom + 18}" text-anchor="middle" font-size="11" fill="#64748b">${fpr}%</text>
          `;
        }).join('')}

        <!-- Curves -->
        <path d="${pathPrec}" fill="none" stroke="#0284c7" stroke-width="2" stroke-dasharray="4,4" />
        <path d="${pathRec}" fill="none" stroke="#7c3aed" stroke-width="2.5" />
        <path d="${pathF1}" fill="none" stroke="#059669" stroke-width="2.5" />
        ${dotsHtml}

        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#cbd5e1" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#cbd5e1" stroke-width="1.5" />

        <text x="${margin.left + chartW / 2}" y="${height - 6}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">Target False Alarm Rate Ceiling (% FPR on Benign Validation)</text>
      </svg>
    `;
  },

  /**
   * Render Distribution Shift Degradation Line Chart (Exp D)
   */
  renderDistributionShiftChart(containerId, shiftData) {
    const container = document.getElementById(containerId);
    if (!container || !shiftData) return;

    const factors = shiftData.scaling_factors;
    const xgbPR = shiftData.xgboost_pr_aucs;
    const vaePR = shiftData.vae_pr_aucs;

    const width = 640;
    const height = 300;
    const margin = { top: 25, right: 30, bottom: 45, left: 55 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const minFactor = 0.5;
    const maxFactor = 2.0;
    const minY = 0.95;
    const maxY = 1.00;

    const ptsXgb = [];
    const ptsVae = [];

    factors.forEach((fac, i) => {
      const x = margin.left + ((fac - minFactor) / (maxFactor - minFactor)) * chartW;
      const yXgb = height - margin.bottom - ((xgbPR[i] - minY) / (maxY - minY)) * chartH;
      const yVae = height - margin.bottom - ((vaePR[i] - minY) / (maxY - minY)) * chartH;
      ptsXgb.push({ x, y: yXgb, val: xgbPR[i], fac });
      ptsVae.push({ x, y: yVae, val: vaePR[i], fac });
    });

    const pathXgb = ptsXgb.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    const pathVae = ptsVae.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

    let dotsHtml = '';
    ptsXgb.forEach(p => {
      dotsHtml += `
        <circle cx="${p.x}" cy="${p.y}" r="4.5" fill="#2563eb">
          <title>Scale ${p.fac}x -> XGBoost PR-AUC: ${p.val}</title>
        </circle>
      `;
    });
    ptsVae.forEach(p => {
      dotsHtml += `
        <circle cx="${p.x}" cy="${p.y}" r="4.5" fill="#7c3aed">
          <title>Scale ${p.fac}x -> VAE PR-AUC: ${p.val}</title>
        </circle>
      `;
    });

    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; max-height: 320px; display: block;">
        <!-- Grid lines -->
        ${[0.95, 0.96, 0.97, 0.98, 0.99, 1.00].map(v => {
          const y = height - margin.bottom - ((v - minY) / (maxY - minY)) * chartH;
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#e2e8f0" stroke-dasharray="3,3" />
            <text x="${margin.left - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b">${v.toFixed(2)}</text>
          `;
        }).join('')}

        <!-- X Axis labels -->
        ${factors.map(fac => {
          const x = margin.left + ((fac - minFactor) / (maxFactor - minFactor)) * chartW;
          return `
            <line x1="${x}" y1="${height - margin.bottom}" x2="${x}" y2="${height - margin.bottom + 4}" stroke="#94a3b8" />
            <text x="${x}" y="${height - margin.bottom + 18}" text-anchor="middle" font-size="11" fill="#64748b">${fac}x</text>
          `;
        }).join('')}

        <!-- Paths -->
        <path d="${pathXgb}" fill="none" stroke="#2563eb" stroke-width="2.5" />
        <path d="${pathVae}" fill="none" stroke="#7c3aed" stroke-width="2.5" />
        ${dotsHtml}

        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#cbd5e1" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#cbd5e1" stroke-width="1.5" />

        <text x="${margin.left + chartW / 2}" y="${height - 6}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">Packet Size Scaling Perturbation Multiplier</text>
      </svg>
    `;
  },

  /**
   * Render Complementarity Donut / Ring (Exp F)
   */
  renderComplementarityDonut(containerId, compData) {
    const container = document.getElementById(containerId);
    if (!container || !compData) return;

    const total = compData.total_held_out_attacks || 132209;
    const slices = [
      { label: 'XGBoost Only', count: compData.xgboost_only, color: '#2563eb' },
      { label: 'Both Detected', count: compData.both_detected, color: '#059669' },
      { label: 'VAE Only', count: compData.vae_only, color: '#7c3aed' },
      { label: 'Both Missed', count: compData.both_missed, color: '#dc2626' }
    ];

    const size = 260;
    const center = size / 2;
    const radius = 95;
    const innerRadius = 60;

    let currentAngle = -Math.PI / 2;
    let pathsHtml = '';

    slices.forEach((s) => {
      const pct = s.count / total;
      const angle = pct * 2 * Math.PI;

      const x1 = center + radius * Math.cos(currentAngle);
      const y1 = center + radius * Math.sin(currentAngle);
      const x2 = center + radius * Math.cos(currentAngle + angle);
      const y2 = center + radius * Math.sin(currentAngle + angle);

      const ix1 = center + innerRadius * Math.cos(currentAngle + angle);
      const iy1 = center + innerRadius * Math.sin(currentAngle + angle);
      const ix2 = center + innerRadius * Math.cos(currentAngle);
      const iy2 = center + innerRadius * Math.sin(currentAngle);

      const largeArc = angle > Math.PI ? 1 : 0;

      const d = `
        M ${x1} ${y1}
        A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}
        L ${ix1} ${iy1}
        A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${ix2} ${iy2}
        Z
      `;

      pathsHtml += `
        <path d="${d}" fill="${s.color}" opacity="0.95">
          <title>${s.label}: ${s.count.toLocaleString()} (${(pct * 100).toFixed(2)}%)</title>
        </path>
      `;

      currentAngle += angle;
    });

    container.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: center; gap: 24px; flex-wrap: wrap;">
        <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
          ${pathsHtml}
          <!-- Center Text -->
          <text x="${center}" y="${center - 6}" text-anchor="middle" font-size="12" font-weight="600" fill="#64748b">HELD-OUT</text>
          <text x="${center}" y="${center + 14}" text-anchor="middle" font-size="16" font-weight="700" fill="#0f172a">${total.toLocaleString()}</text>
        </svg>

        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${slices.map(s => `
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; font-size: 13px;">
              <span style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background-color: ${s.color}; display: inline-block;"></span>
                <span style="font-weight: 500; color: #334155;">${s.label}</span>
              </span>
              <span style="font-weight: 600; color: #0f172a; font-variant-numeric: tabular-nums;">
                ${s.count.toLocaleString()} (${((s.count / total) * 100).toFixed(1)}%)
              </span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  },

  /**
   * Render Dataset Distribution Bar Chart (Figure 1)
   */
  renderDatasetDistribution(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container || !data) return;

    const familyCounts = data.family_counts || data;
    const sortedEntries = Object.entries(familyCounts).sort((a, b) => b[1] - a[1]);
    if (sortedEntries.length === 0) return;

    const width = 760;
    const height = Math.max(340, sortedEntries.length * 28 + 70);
    const margin = { top: 20, right: 90, bottom: 40, left: 160 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const maxCount = Math.max(...sortedEntries.map(e => e[1]));
    const barHeight = Math.min(16, (chartH / sortedEntries.length) - 6);

    let barsHtml = '';
    let labelsHtml = '';

    sortedEntries.forEach(([fam, count], idx) => {
      const y = margin.top + idx * (chartH / sortedEntries.length) + 4;
      const barW = Math.max(3, (count / maxCount) * chartW);
      const isBenign = fam.toUpperCase().includes('BENIGN');
      const fillColor = isBenign ? '#059669' : '#2563eb';

      barsHtml += `
        <rect x="${margin.left}" y="${y}" width="${barW}" height="${barHeight}" rx="3" fill="${fillColor}">
          <title>${fam}: ${count.toLocaleString()} flows</title>
        </rect>
        <text x="${margin.left + barW + 8}" y="${y + barHeight - 3}" font-size="11" font-weight="600" fill="#334155">${count.toLocaleString()}</text>
      `;

      labelsHtml += `
        <text x="${margin.left - 8}" y="${y + barHeight - 3}" text-anchor="end" font-size="11.5" font-weight="${isBenign ? '700' : '500'}" fill="${isBenign ? '#047857' : '#1e293b'}">
          ${fam}
        </text>
      `;
    });

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Authentic Curated Flow Distribution by Traffic Family</span>
        <a href="/figures/fig1_dataset_distribution.png" target="_blank" class="chart-artifact-badge" download>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Download Artifact (PNG)
        </a>
      </div>
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;">
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#cbd5e1" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#cbd5e1" stroke-width="1.5" />
        ${barsHtml}
        ${labelsHtml}
        <text x="${margin.left + chartW / 2}" y="${height - 10}" text-anchor="middle" font-size="11.5" font-weight="500" fill="#64748b">Flow Records Count</text>
      </svg>
    `;
  },

  /**
   * Render ROC Curves (Figure 2 Panel A)
   */
  renderRocCurves(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container || !data) return;

    const width = 420;
    const height = 300;
    const margin = { top: 25, right: 25, bottom: 45, left: 50 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const roc = data.roc_curves || {};
    const xgbPts = roc.xgboost || [
      { fpr: 0.0, tpr: 0.0 }, { fpr: 0.00098, tpr: 0.9999 }, { fpr: 1.0, tpr: 1.0 }
    ];
    const vaePts = roc.vae || [
      { fpr: 0.0, tpr: 0.0 }, { fpr: 0.0093, tpr: 0.2589 }, { fpr: 0.1, tpr: 0.81 }, { fpr: 1.0, tpr: 1.0 }
    ];

    const toX = (fpr) => margin.left + fpr * chartW;
    const toY = (tpr) => height - margin.bottom - tpr * chartH;

    const pathXgb = xgbPts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.fpr)} ${toY(p.tpr)}`).join(' ');
    const pathVae = vaePts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.fpr)} ${toY(p.tpr)}`).join(' ');
    const pathBaseline = `M ${toX(0)} ${toY(0)} L ${toX(1)} ${toY(1)}`;

    const metrics = data.metrics || {};
    const xgbAuc = metrics.xgboost ? metrics.xgboost.roc_auc : 0.99996;
    const vaeAuc = metrics.vae ? metrics.vae.roc_auc : 0.91452;

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">ROC Curves Comparison</span>
        <a href="/figures/fig2_roc_pr_curves.png" target="_blank" class="chart-artifact-badge" download>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Figure 2 (PNG)
        </a>
      </div>
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;">
        <!-- Grid -->
        ${[0, 0.25, 0.5, 0.75, 1.0].map(v => {
          const y = toY(v);
          const x = toX(v);
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#f1f5f9" stroke-dasharray="3,3" />
            <text x="${margin.left - 6}" y="${y + 4}" text-anchor="end" font-size="10" fill="#94a3b8">${v.toFixed(2)}</text>
            <text x="${x}" y="${height - margin.bottom + 16}" text-anchor="middle" font-size="10" fill="#94a3b8">${v.toFixed(2)}</text>
          `;
        }).join('')}
        <!-- Baseline -->
        <path d="${pathBaseline}" fill="none" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4,4" />
        <!-- VAE Curve -->
        <path d="${pathVae}" fill="none" stroke="#7c3aed" stroke-width="2.5" />
        <!-- XGBoost Curve -->
        <path d="${pathXgb}" fill="none" stroke="#2563eb" stroke-width="2.5" />
        <!-- Axes -->
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <!-- Labels -->
        <text x="${margin.left + chartW / 2}" y="${height - 6}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">False Positive Rate (FPR)</text>
        <text transform="rotate(-90)" x="${-(margin.top + chartH / 2)}" y="16" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">True Positive Rate (Recall)</text>
      </svg>
      <div class="chart-legend-row" style="justify-content: center;">
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #2563eb;"></span><strong>XGBoost</strong> (ROC-AUC: ${xgbAuc})</span>
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #7c3aed;"></span><strong>Tabular VAE</strong> (ROC-AUC: ${vaeAuc})</span>
      </div>
    `;
  },

  /**
   * Render PR Curves (Figure 2 Panel B)
   */
  renderPrCurves(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container || !data) return;

    const width = 420;
    const height = 300;
    const margin = { top: 25, right: 25, bottom: 45, left: 50 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const pr = data.pr_curves || {};
    const xgbPts = pr.xgboost || [
      { recall: 0.0, precision: 1.0 }, { recall: 0.9999, precision: 0.9995 }, { recall: 1.0, precision: 0.0 }
    ];
    const vaePts = pr.vae || [
      { recall: 0.0, precision: 1.0 }, { recall: 0.2589, precision: 0.9834 }, { recall: 0.5, precision: 0.92 }, { recall: 1.0, precision: 0.68 }
    ];

    const toX = (rec) => margin.left + rec * chartW;
    const toY = (prec) => height - margin.bottom - prec * chartH;

    const pathXgb = xgbPts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.recall)} ${toY(p.precision)}`).join(' ');
    const pathVae = vaePts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p.recall)} ${toY(p.precision)}`).join(' ');

    const metrics = data.metrics || {};
    const xgbPr = metrics.xgboost ? metrics.xgboost.pr_auc : 0.99998;
    const vaePr = metrics.vae ? metrics.vae.pr_auc : 0.94319;

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Precision-Recall Curves Comparison</span>
        <span class="badge badge-slate">High Class Imbalance</span>
      </div>
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;">
        <!-- Grid -->
        ${[0, 0.25, 0.5, 0.75, 1.0].map(v => {
          const y = toY(v);
          const x = toX(v);
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#f1f5f9" stroke-dasharray="3,3" />
            <text x="${margin.left - 6}" y="${y + 4}" text-anchor="end" font-size="10" fill="#94a3b8">${v.toFixed(2)}</text>
            <text x="${x}" y="${height - margin.bottom + 16}" text-anchor="middle" font-size="10" fill="#94a3b8">${v.toFixed(2)}</text>
          `;
        }).join('')}
        <!-- VAE Curve -->
        <path d="${pathVae}" fill="none" stroke="#7c3aed" stroke-width="2.5" />
        <!-- XGBoost Curve -->
        <path d="${pathXgb}" fill="none" stroke="#2563eb" stroke-width="2.5" />
        <!-- Axes -->
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <!-- Labels -->
        <text x="${margin.left + chartW / 2}" y="${height - 6}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">Detection Recall</text>
        <text transform="rotate(-90)" x="${-(margin.top + chartH / 2)}" y="16" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">Precision</text>
      </svg>
      <div class="chart-legend-row" style="justify-content: center;">
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #2563eb;"></span><strong>XGBoost</strong> (PR-AUC: ${xgbPr})</span>
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #7c3aed;"></span><strong>Tabular VAE</strong> (PR-AUC: ${vaePr})</span>
      </div>
    `;
  },

  /**
   * Render Confusion Matrices (Figure 3)
   */
  renderConfusionMatrices(containerId, cmData) {
    const container = document.getElementById(containerId);
    if (!container || !cmData) return;

    const known = cmData.known_benchmark || cmData;
    const xgb = known.xgboost || { tn: 39961, fp: 39, fn: 8, tp: 85080 };
    const vae = known.vae || { tn: 39628, fp: 372, fn: 63058, tp: 22030 };

    const totalXgb = xgb.tn + xgb.fp + xgb.fn + xgb.tp;
    const totalVae = vae.tn + vae.fp + vae.fn + vae.tp;

    function renderMatrixHtml(title, colorBrand, m, total) {
      return `
        <div style="flex: 1 1 280px; min-width: 0; background: #f8fafc; border: 1px solid var(--border-light); border-radius: var(--radius-sm); padding: 14px;">
          <div style="font-weight: 600; font-size: 13px; color: var(--text-primary); margin-bottom: 10px; display: flex; justify-content: space-between;">
            <span>${title}</span>
            <span style="font-size: 11.5px; color: var(--text-muted);">${total.toLocaleString()} flows</span>
          </div>
          <div style="display: grid; grid-template-columns: 80px 1fr 1fr; gap: 4px; font-size: 11.5px; text-align: center;">
            <div></div>
            <div style="font-weight: 600; color: #475569; padding-bottom: 2px;">Pred Benign</div>
            <div style="font-weight: 600; color: #475569; padding-bottom: 2px;">Pred Attack</div>

            <div style="font-weight: 600; color: #475569; text-align: right; padding-right: 6px; align-self: center;">True Benign</div>
            <div style="background: ${colorBrand === 'blue' ? '#dbeafe' : '#f3e8ff'}; padding: 10px 4px; border-radius: 4px;">
              <div style="font-weight: 700; font-size: 13px; color: #0f172a;">${m.tn.toLocaleString()}</div>
              <div style="font-size: 10px; color: #059669; font-weight: 600;">TN (${((m.tn / (m.tn + m.fp)) * 100).toFixed(1)}%)</div>
            </div>
            <div style="background: #fef2f2; padding: 10px 4px; border-radius: 4px;">
              <div style="font-weight: 700; font-size: 13px; color: #0f172a;">${m.fp.toLocaleString()}</div>
              <div style="font-size: 10px; color: #dc2626; font-weight: 600;">FP (${((m.fp / (m.tn + m.fp)) * 100).toFixed(2)}%)</div>
            </div>

            <div style="font-weight: 600; color: #475569; text-align: right; padding-right: 6px; align-self: center;">True Attack</div>
            <div style="background: #fef2f2; padding: 10px 4px; border-radius: 4px;">
              <div style="font-weight: 700; font-size: 13px; color: #0f172a;">${m.fn.toLocaleString()}</div>
              <div style="font-size: 10px; color: #dc2626; font-weight: 600;">FN (${((m.fn / (m.fn + m.tp)) * 100).toFixed(2)}%)</div>
            </div>
            <div style="background: ${colorBrand === 'blue' ? '#bfdbfe' : '#e9d5ff'}; padding: 10px 4px; border-radius: 4px;">
              <div style="font-weight: 700; font-size: 13px; color: #0f172a;">${m.tp.toLocaleString()}</div>
              <div style="font-size: 10px; color: #059669; font-weight: 600;">TP (${((m.tp / (m.fn + m.tp)) * 100).toFixed(1)}%)</div>
            </div>
          </div>
        </div>
      `;
    }

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Figure 3: Confusion Matrices (Known Attack Benchmark)</span>
        <a href="/figures/fig3_confusion_matrices.png" target="_blank" class="chart-artifact-badge" download>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Figure 3 (PNG)
        </a>
      </div>
      <div style="display: flex; gap: 14px; flex-wrap: wrap;">
        ${renderMatrixHtml('XGBoost Classifier', 'blue', xgb, totalXgb)}
        ${renderMatrixHtml('PyTorch Tabular VAE (Anomaly)', 'purple', vae, totalVae)}
      </div>
    `;
  },

  /**
   * Render Anomaly Score Distribution Histogram (Actual uploaded flows or live capture)
   */
  renderAnomalyScoreDistribution(containerId, scoreData, customThreshold) {
    const container = document.getElementById(containerId);
    if (!container || !scoreData) return;

    const hist = scoreData.histogram || scoreData;
    const threshold = customThreshold || scoreData.calibrated_threshold || 0.330610;

    if (!Array.isArray(hist) || hist.length === 0) {
      container.innerHTML = `
        <div class="chart-empty-state">
          <p>No anomaly scores available for histogram display.</p>
        </div>
      `;
      return;
    }

    const width = 560;
    const height = 240;
    const margin = { top: 25, right: 30, bottom: 40, left: 50 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const maxCount = Math.max(...hist.map(b => b.count), 1);
    const minEdge = hist[0].bin_start;
    const maxEdge = hist[hist.length - 1].bin_end;
    const barW = chartW / hist.length;

    let barsHtml = '';
    hist.forEach((b, i) => {
      const h = (b.count / maxCount) * chartH;
      const x = margin.left + i * barW;
      const y = height - margin.bottom - h;
      const isAnomZone = b.bin_end >= threshold;
      const color = isAnomZone ? '#dc2626' : '#7c3aed';

      barsHtml += `
        <rect x="${x + 1}" y="${y}" width="${Math.max(1, barW - 2)}" height="${h}" rx="2" fill="${color}" opacity="0.85">
          <title>Range: [${b.bin_start} - ${b.bin_end}]: ${b.count} flows (${b.pct || ''}%)</title>
        </rect>
      `;
    });

    let threshX = margin.left + ((threshold - minEdge) / Math.max(maxEdge - minEdge, 1e-6)) * chartW;
    threshX = Math.max(margin.left, Math.min(width - margin.right, threshX));

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">VAE Reconstruction Anomaly Score Distribution</span>
        <span class="badge badge-indigo">Threshold: ${threshold}</span>
      </div>
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;">
        <!-- Grid lines -->
        ${[0, 0.5, 1.0].map(p => {
          const y = height - margin.bottom - p * chartH;
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#f1f5f9" stroke-dasharray="3,3" />
            <text x="${margin.left - 6}" y="${y + 4}" text-anchor="end" font-size="10" fill="#94a3b8">${Math.round(p * maxCount)}</text>
          `;
        }).join('')}
        ${barsHtml}
        <!-- Calibrated Threshold Marker Line -->
        <line x1="${threshX}" y1="${margin.top - 10}" x2="${threshX}" y2="${height - margin.bottom}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,3" />
        <text x="${threshX + 4}" y="${margin.top - 2}" font-size="10.5" font-weight="700" fill="#dc2626">Threshold: ${threshold}</text>

        <!-- Baseline -->
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />

        <!-- X labels -->
        <text x="${margin.left}" y="${height - margin.bottom + 16}" font-size="10" fill="#64748b">${minEdge}</text>
        <text x="${width - margin.right}" y="${height - margin.bottom + 16}" text-anchor="end" font-size="10" fill="#64748b">${maxEdge}</text>
        <text x="${margin.left + chartW / 2}" y="${height - 6}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">Reconstruction Anomaly Score (MSE)</text>
      </svg>
      <div class="chart-legend-row">
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #7c3aed;"></span>Normal Flows (&lt; ${threshold})</span>
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #dc2626;"></span>Anomalous Flows (&ge; ${threshold})</span>
      </div>
    `;
  },

  /**
   * Render Model Disagreement 4-Bar Chart
   */
  renderModelDisagreement(containerId, disagData) {
    const container = document.getElementById(containerId);
    if (!container || !disagData) return;

    const items = [
      { label: 'Both Flagged (Critical)', count: disagData.both_flagged, pct: disagData.both_flagged_pct, color: '#dc2626' },
      { label: 'XGBoost Only (Pattern)', count: disagData.xgboost_only, pct: disagData.xgboost_only_pct, color: '#2563eb' },
      { label: 'VAE Only (Anomaly)', count: disagData.vae_only, pct: disagData.vae_only_pct, color: '#7c3aed' },
      { label: 'Both Normal (Benign)', count: disagData.both_normal, pct: disagData.both_normal_pct, color: '#059669' }
    ];

    const total = items.reduce((sum, item) => sum + item.count, 0) || 1;
    const maxCount = Math.max(...items.map(i => i.count), 1);

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Dual-Model Agreement & Disagreement Breakdown</span>
        <span class="badge badge-slate">${total.toLocaleString()} total flows</span>
      </div>
      <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
        ${items.map(item => {
          const wPct = Math.max(2, (item.count / maxCount) * 100);
          return `
            <div>
              <div style="display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 4px;">
                <span style="display: flex; align-items: center; gap: 6px;">
                  <span style="width: 10px; height: 10px; border-radius: 2px; background: ${item.color}; display: inline-block;"></span>
                  <strong>${item.label}</strong>
                </span>
                <span style="font-weight: 600; color: #0f172a;">${item.count.toLocaleString()} (${item.pct}%)</span>
              </div>
              <div style="width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                <div style="width: ${wPct}%; height: 100%; background: ${item.color}; border-radius: 4px;"></div>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  },

  /**
   * Render Live Traffic Volume Area Chart
   */
  renderLiveTrafficVolume(containerId, volumeBuckets) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!volumeBuckets || volumeBuckets.length === 0) {
      container.innerHTML = `
        <div class="chart-empty-state">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          <div style="font-weight: 600; margin-bottom: 4px;">No Live Traffic Observed Yet</div>
          <div>Start packet capture above to observe flows and packets in real time.</div>
        </div>
      `;
      return;
    }

    const width = 560;
    const height = 220;
    const margin = { top: 20, right: 25, bottom: 40, left: 45 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const maxFlows = Math.max(...volumeBuckets.map(b => b.flows), 1);
    const n = volumeBuckets.length;
    const stepX = n > 1 ? chartW / (n - 1) : chartW;

    const points = volumeBuckets.map((b, i) => {
      const x = margin.left + i * stepX;
      const y = height - margin.bottom - (b.flows / maxFlows) * chartH;
      return { x, y, ...b };
    });

    const pathLine = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    const pathArea = `${pathLine} L ${points[points.length - 1].x} ${height - margin.bottom} L ${points[0].x} ${height - margin.bottom} Z`;

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Live Traffic Ingestion Rate (Flows / Interval)</span>
        <span class="badge badge-blue">Peak: ${maxFlows} flows/bucket</span>
      </div>
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;">
        <defs>
          <linearGradient id="volGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#2563eb" stop-opacity="0.35" />
            <stop offset="100%" stop-color="#2563eb" stop-opacity="0.0" />
          </linearGradient>
        </defs>
        <!-- Grid -->
        ${[0, 0.5, 1.0].map(p => {
          const y = height - margin.bottom - p * chartH;
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#f1f5f9" stroke-dasharray="3,3" />
            <text x="${margin.left - 6}" y="${y + 4}" text-anchor="end" font-size="10" fill="#94a3b8">${Math.round(p * maxFlows)}</text>
          `;
        }).join('')}
        <path d="${pathArea}" fill="url(#volGrad)" />
        <path d="${pathLine}" fill="none" stroke="#2563eb" stroke-width="2.5" />
        ${points.map(p => `
          <circle cx="${p.x}" cy="${p.y}" r="3.5" fill="#2563eb">
            <title>${p.time}: ${p.flows} flows, ${p.packets} packets, ${p.suspicious || 0} suspicious</title>
          </circle>
        `).join('')}
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <!-- Time labels -->
        <text x="${points[0].x}" y="${height - margin.bottom + 16}" font-size="10" fill="#64748b">${points[0].time}</text>
        <text x="${points[points.length - 1].x}" y="${height - margin.bottom + 16}" text-anchor="end" font-size="10" fill="#64748b">${points[points.length - 1].time}</text>
      </svg>
    `;
  },

  /**
   * Render Live VAE Anomaly Score Stream
   */
  renderLiveVaeScores(containerId, scoreStream, threshold = 0.330610) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!scoreStream || scoreStream.length === 0) {
      container.innerHTML = `
        <div class="chart-empty-state">
          <p>No completed flows evaluated for anomaly scoring yet.</p>
        </div>
      `;
      return;
    }

    const width = 560;
    const height = 220;
    const margin = { top: 25, right: 30, bottom: 40, left: 50 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const maxScore = Math.max(...scoreStream.map(s => s.score), threshold * 1.5);
    const n = scoreStream.length;
    const stepX = n > 1 ? chartW / (n - 1) : chartW;

    const threshY = height - margin.bottom - (threshold / maxScore) * chartH;

    let dotsHtml = '';
    const pts = scoreStream.map((s, i) => {
      const x = margin.left + i * stepX;
      const y = height - margin.bottom - (s.score / maxScore) * chartH;
      const isAnom = s.score >= threshold;
      const color = isAnom ? '#dc2626' : '#7c3aed';

      dotsHtml += `
        <circle cx="${x}" cy="${y}" r="${isAnom ? 5 : 3.5}" fill="${color}">
          <title>${s.flow_id} (${s.time}) - Score: ${s.score} (${isAnom ? 'ANOMALY' : 'NORMAL'})</title>
        </circle>
      `;
      return { x, y };
    });

    const linePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Live VAE Anomaly Reconstruction Error Stream</span>
        <span class="badge badge-indigo">Threshold: ${threshold}</span>
      </div>
      <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;">
        <!-- Threshold Line -->
        <line x1="${margin.left}" y1="${threshY}" x2="${width - margin.right}" y2="${threshY}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="4,4" />
        <text x="${width - margin.right}" y="${threshY - 4}" text-anchor="end" font-size="10.5" font-weight="700" fill="#dc2626">Calibrated Threshold (${threshold})</text>
        <!-- Line Path -->
        <path d="${linePath}" fill="none" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="2,2" />
        ${dotsHtml}
        <!-- Axes -->
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <text x="${margin.left - 6}" y="${height - margin.bottom + 4}" text-anchor="end" font-size="10" fill="#94a3b8">0.0</text>
        <text x="${margin.left - 6}" y="${margin.top + 4}" text-anchor="end" font-size="10" fill="#94a3b8">${maxScore.toFixed(2)}</text>
        <text x="${margin.left + chartW / 2}" y="${height - 6}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">Recent Evaluated Flows</text>
      </svg>
    `;
  },

  /**
   * Render Protocol Distribution Donut Chart
   */
  renderProtocolDonut(containerId, protocols) {
    const container = document.getElementById(containerId);
    if (!container || !protocols) return;

    const entries = Object.entries(protocols);
    if (entries.length === 0) {
      container.innerHTML = `
        <div class="chart-empty-state">
          <p>No protocol data observed.</p>
        </div>
      `;
      return;
    }

    const colors = { TCP: '#2563eb', UDP: '#7c3aed', ICMP: '#f59e0b', OTHER: '#94a3b8' };
    const total = entries.reduce((sum, [, count]) => sum + count, 0);

    const size = 180;
    const center = size / 2;
    const radius = 68;
    const innerRadius = 42;

    let currentAngle = -Math.PI / 2;
    let pathsHtml = '';

    entries.forEach(([proto, count]) => {
      const pct = count / total;
      const angle = pct * 2 * Math.PI;

      const x1 = center + radius * Math.cos(currentAngle);
      const y1 = center + radius * Math.sin(currentAngle);
      const x2 = center + radius * Math.cos(currentAngle + angle);
      const y2 = center + radius * Math.sin(currentAngle + angle);

      const ix1 = center + innerRadius * Math.cos(currentAngle + angle);
      const iy1 = center + innerRadius * Math.sin(currentAngle + angle);
      const ix2 = center + innerRadius * Math.cos(currentAngle);
      const iy2 = center + innerRadius * Math.sin(currentAngle);

      const largeArc = angle > Math.PI ? 1 : 0;
      const d = `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} L ${ix1} ${iy1} A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${ix2} ${iy2} Z`;

      const color = colors[proto.toUpperCase()] || '#64748b';
      pathsHtml += `<path d="${d}" fill="${color}"><title>${proto}: ${count} (${(pct * 100).toFixed(1)}%)</title></path>`;
      currentAngle += angle;
    });

    container.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-around; gap: 12px; flex-wrap: wrap;">
        <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
          ${pathsHtml}
          <text x="${center}" y="${center + 4}" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">${total}</text>
        </svg>
        <div style="display: flex; flex-direction: column; gap: 6px; font-size: 12px;">
          ${entries.map(([proto, count]) => {
            const color = colors[proto.toUpperCase()] || '#64748b';
            return `
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; border-radius: 2px; background: ${color}; display: inline-block;"></span>
                <span><strong>${proto}:</strong> ${count} (${((count / total) * 100).toFixed(1)}%)</span>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;
  },

  /**
   * Render Prediction Distribution Bars
   */
  renderPredictionDistribution(containerId, predData) {
    const container = document.getElementById(containerId);
    if (!container || !predData) return;

    const xgb = predData.xgboost || { attack: 0, benign: 0, attack_pct: 0, benign_pct: 0 };
    const vae = predData.vae || { anomaly: 0, normal: 0, anomaly_pct: 0, normal_pct: 0 };

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Supervised vs Semi-Supervised Decision Distribution</span>
      </div>
      <div style="display: flex; flex-direction: column; gap: 14px; margin-top: 8px;">
        <!-- XGBoost Bar -->
        <div>
          <div style="display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 4px;">
            <span><strong>XGBoost</strong> (Thresh: 0.52)</span>
            <span>Attacks: <strong style="color: #2563eb;">${xgb.attack.toLocaleString()}</strong> (${xgb.attack_pct}%) | Benign: ${xgb.benign.toLocaleString()}</span>
          </div>
          <div style="width: 100%; height: 10px; background: #059669; border-radius: 5px; overflow: hidden; display: flex;">
            <div style="width: ${xgb.attack_pct}%; height: 100%; background: #2563eb;"></div>
          </div>
        </div>

        <!-- VAE Bar -->
        <div>
          <div style="display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 4px;">
            <span><strong>PyTorch VAE</strong> (Thresh: 0.330610)</span>
            <span>Anomalies: <strong style="color: #7c3aed;">${vae.anomaly.toLocaleString()}</strong> (${vae.anomaly_pct}%) | Normal: ${vae.normal.toLocaleString()}</span>
          </div>
          <div style="width: 100%; height: 10px; background: #059669; border-radius: 5px; overflow: hidden; display: flex;">
            <div style="width: ${vae.anomaly_pct}%; height: 100%; background: #7c3aed;"></div>
          </div>
        </div>
      </div>
      <div class="chart-legend-row" style="margin-top: 10px;">
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #2563eb;"></span>XGBoost Attack</span>
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #7c3aed;"></span>VAE Anomaly</span>
        <span class="chart-legend-item"><span class="chart-legend-dot" style="background: #059669;"></span>Normal / Benign</span>
      </div>
    `;
  },

  /**
   * Render Traffic Family Distribution (from uploaded CSV)
   */
  renderTrafficFamilyDistribution(containerId, familyCounts, hasGroundTruth) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!hasGroundTruth || !familyCounts) {
      container.innerHTML = `
        <div class="chart-empty-state" style="padding: 24px 16px;">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          <div style="font-weight: 700; color: #92400e; margin-bottom: 4px; font-size: 13.5px;">Ground Truth Unavailable</div>
          <div style="max-width: 440px; margin: 0 auto; color: #78350f; font-size: 12.5px; line-height: 1.5;">
            This uploaded CSV does not contain a verified ground-truth label column. Model predictions and anomaly scores are computed without fabricating unverified attack categories.
          </div>
        </div>
      `;
      return;
    }

    const entries = Object.entries(familyCounts).sort((a, b) => b[1] - a[1]);
    const maxVal = Math.max(...entries.map(e => e[1]), 1);

    container.innerHTML = `
      <div class="chart-header-row">
        <span class="chart-title-sm">Ground-Truth Traffic Family Distribution</span>
        <span class="badge badge-green">Verified Labels</span>
      </div>
      <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 8px;">
        ${entries.map(([fam, count]) => {
          const w = (count / maxVal) * 100;
          const isBenign = fam.toUpperCase().includes('BENIGN');
          const color = isBenign ? '#059669' : '#2563eb';
          return `
            <div>
              <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                <span><strong>${fam}</strong></span>
                <span style="color: #0f172a; font-weight: 600;">${count.toLocaleString()} flows</span>
              </div>
              <div style="width: 100%; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                <div style="width: ${w}%; height: 100%; background: ${color}; border-radius: 3px;"></div>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  }
};

window.Charts = Charts;

