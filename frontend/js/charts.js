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
  }
};

window.Charts = Charts;
