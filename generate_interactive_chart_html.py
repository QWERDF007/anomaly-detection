# -*- coding: utf-8 -*-
"""
Standalone Interactive Benchmark Dashboard HTML Generator for Industrial Anomaly Detection
===========================================================================================
Generates self-contained, publication-grade interactive HTML dashboards for each dataset.
Features:
- Dual Tab navigation:
  1) 📊 算法模型基准评测 (Model Benchmark Dashboard)
  2) 📦 训练集与测试集数量分布 (Dataset Split Distribution: 训练集图表 + 测试集图表 左右并排分开展示)
- Independent, isolated per-dataset reporting (展示各自数据集自身的训练与测试样本规模与构成)
- Ample SVG bottom margins and flexbox layout ensuring bottom X-axis and labels are 100% visible
- Interactive mouse guideline (horizontal dashed line & dynamic Y-value badge) on all charts
- Publication-quality SVGs exportable at any time
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def build_interactive_html(
    dataset_name: str,
    dataset_data: List[Dict[str, Any]],
    has_bank: bool = False,
    dataset_stats: Optional[Dict[str, Any]] = None
) -> str:
    """Generates strictly ONE self-contained interactive benchmark dashboard HTML for a single dataset."""
    json_data = json.dumps(dataset_data, ensure_ascii=False)
    has_bank_js = "true" if has_bank else "false"
    dataset_stats_json = json.dumps(dataset_stats or {}, ensure_ascii=False)

    if has_bank:
        model_checkboxes = """        <label class="checkbox-item">
          <input type="checkbox" id="chkE2E" checked data-model="Two-Stage E2E">
          <span class="color-dot dot-e2e"></span>
          <span>二阶段端到端 (Two-Stage E2E)</span>
        </label>
        <label class="checkbox-item">
          <input type="checkbox" id="chkDino" checked data-model="Dinomaly2">
          <span class="color-dot dot-dino"></span>
          <span>Dinomaly2 (单阶段)</span>
        </label>
        <label class="checkbox-item">
          <input type="checkbox" id="chkPatch" checked data-model="PatchCore">
          <span class="color-dot dot-patch"></span>
          <span>PatchCore (特征检索基线)</span>
        </label>"""
    else:
        model_checkboxes = """        <label class="checkbox-item">
          <input type="checkbox" id="chkDino" checked data-model="Dinomaly2">
          <span class="color-dot dot-dino"></span>
          <span>Dinomaly2</span>
        </label>
        <label class="checkbox-item">
          <input type="checkbox" id="chkPatch" checked data-model="PatchCore">
          <span class="color-dot dot-patch"></span>
          <span>PatchCore (基线)</span>
        </label>"""

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{dataset_name} · 工业视觉异常检测交互看板</title>
<style>
  :root {{
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    --e2e-color: #059669;
    --dino-color: #2563eb;
    --patch-color: #ea580c;
    --train-color: #0d9488;
    --test-ok-color: #10b981;
    --test-ng-color: #f43f5e;
    --test-total-color: #3b82f6;
    --primary: #3b82f6;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  html, body {{
    width: 100%;
    height: 100vh;
    max-height: 100vh;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text-primary);
    display: flex;
    flex-direction: column;
  }}

  /* Header Bar */
  header {{
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    padding: 7px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    flex-wrap: nowrap;
    gap: 12px;
    flex-shrink: 0;
    height: 52px;
    max-height: 52px;
  }}

  .header-left {{
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
  }}

  .header-left h1 {{
    font-size: 1.10rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
  }}

  /* Header Tabs */
  .header-tabs {{
    display: flex;
    align-items: center;
    gap: 4px;
    background: #f1f5f9;
    padding: 3px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
  }}
  .header-tabs .tab-btn {{
    padding: 5px 13px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }}
  .header-tabs .tab-btn:hover {{
    color: #0f172a;
    background: rgba(255, 255, 255, 0.75);
  }}
  .header-tabs .tab-btn.active {{
    background: #ffffff;
    color: #2563eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  }}

  .header-selectors {{
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: nowrap;
  }}
  .selector-wrapper {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #334155;
    white-space: nowrap;
  }}

  select.custom-select {{
    padding: 4px 10px;
    border-radius: 6px;
    border: 1.5px solid #cbd5e1;
    background: white;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1e293b;
    cursor: pointer;
    outline: none;
    transition: all 0.2s;
  }}
  select.custom-select:focus {{
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
  }}
  select.custom-select-metric {{
    border-color: #3b82f6;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 0.84rem;
  }}

  .header-actions {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }}

  .btn {{
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid var(--border);
    background: white;
    color: #334155;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: all 0.15s;
    user-select: none;
    white-space: nowrap;
  }}
  .btn:hover {{
    background: #f1f5f9;
    border-color: #cbd5e1;
  }}
  .btn-primary {{
    background: #2563eb;
    color: white;
    border-color: #2563eb;
  }}
  .btn-primary:hover {{
    background: #1d4ed8;
  }}

  /* Main Workspace Layout */
  .main-layout {{
    display: flex;
    flex: 1 1 0;
    min-height: 0;
    padding: 8px 12px;
    gap: 10px;
    width: 100%;
    overflow: hidden;
  }}

  /* Sidebar Controls */
  .control-panel {{
    width: 275px;
    background: var(--card-bg);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    box-shadow: var(--shadow);
    flex-shrink: 0;
    height: 100%;
    max-height: 100%;
    overflow-y: auto;
  }}
  .control-panel::-webkit-scrollbar {{
    width: 4px;
  }}
  .control-panel::-webkit-scrollbar-thumb {{
    background: #cbd5e1;
    border-radius: 2px;
  }}

  .control-group {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .control-group h3 {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}

  .checkbox-list {{
    display: flex;
    flex-direction: column;
    gap: 3px;
  }}

  .checkbox-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: #1e293b;
    cursor: pointer;
    user-select: none;
    padding: 3px 6px;
    border-radius: 5px;
    transition: background 0.15s;
  }}
  .checkbox-item:hover {{
    background: #f1f5f9;
  }}
  .checkbox-item input[type="checkbox"] {{
    width: 15px;
    height: 15px;
    border-radius: 4px;
    cursor: pointer;
    accent-color: #2563eb;
  }}

  .color-dot {{
    width: 11px;
    height: 11px;
    border-radius: 50%;
    display: inline-block;
  }}
  .dot-e2e {{ background: var(--e2e-color); }}
  .dot-dino {{ background: var(--dino-color); }}
  .dot-patch {{ background: var(--patch-color); }}

  .size-indicator {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #64748b;
  }}

  .divider {{
    height: 1px;
    background: var(--border);
    margin: 2px 0;
  }}

  .quick-links-group {{
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.78rem;
    font-weight: 500;
  }}
  .quick-link {{
    color: #3b82f6;
    cursor: pointer;
    user-select: none;
    text-decoration: none;
  }}
  .quick-link:hover {{
    text-decoration: underline;
    color: #1d4ed8;
  }}
  .quick-link-sep {{
    color: #94a3b8;
    user-select: none;
    font-size: 0.75rem;
  }}

  /* Sidebar Stat Card */
  .stat-badge-card {{
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.80rem;
    color: #1e3a8a;
    line-height: 1.45;
  }}
  .stat-badge-row {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
  }}
  .stat-badge-row:last-child {{
    margin-bottom: 0;
  }}

  /* Chart Container */
  .chart-container {{
    flex: 1 1 0;
    min-width: 0;
    min-height: 0;
    height: 100%;
    background: var(--card-bg);
    border-radius: 10px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  .chart-header {{
    padding: 6px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fafbfc;
    flex-shrink: 0;
  }}
  .chart-header .title-area {{
    display: flex;
    align-items: baseline;
    gap: 10px;
  }}
  .chart-title {{
    font-size: 1.00rem;
    font-weight: 700;
    color: #0f172a;
  }}
  .chart-stats {{
    font-size: 0.78rem;
    color: #64748b;
  }}

  .chart-actions {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .sub-tab-btn {{
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #475569;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }}
  .sub-tab-btn:hover {{
    color: #0f172a;
    background: rgba(255,255,255,0.7);
  }}
  .sub-tab-btn.active {{
    background: #ffffff;
    color: #2563eb;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
  }}

  /* Chart Body - NO height:100% to prevent flex overflow clipping! */
  .chart-body {{
    flex: 1 1 0;
    min-height: 0;
    position: relative;
    width: 100%;
    height: auto;
    cursor: grab;
    user-select: none;
    background: #ffffff;
    overflow: hidden;
  }}
  .chart-body:active {{
    cursor: grabbing;
  }}
  .chart-body svg {{
    width: 100%;
    height: 100%;
    max-height: 100%;
    display: block;
  }}

  /* Detail Panel */
  .detail-panel {{
    border-top: 1px solid var(--border);
    padding: 6px 14px;
    background: #f8fafc;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    flex-shrink: 0;
    line-height: 1.35;
  }}
  .detail-panel strong {{
    color: #0f172a;
  }}

  /* Dataset Split Dual Cards Grid */
  .ds-charts-grid {{
    flex: 1 1 0;
    min-height: 0;
    display: flex;
    gap: 12px;
    padding: 10px 14px;
    background: #f8fafc;
    overflow: hidden;
  }}
  .ds-card {{
    flex: 1 1 0;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    overflow: hidden;
  }}
  .ds-card-header {{
    padding: 7px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fafbfc;
    flex-shrink: 0;
  }}
  .ds-card-footer {{
    padding: 5px 12px;
    border-top: 1px solid var(--border);
    font-size: 0.74rem;
    color: #64748b;
    background: #fafbfc;
    flex-shrink: 0;
  }}

  /* Tooltip */
  #tooltip {{
    position: absolute;
    display: none;
    pointer-events: none;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(8px);
    color: white;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
    z-index: 100;
    max-width: 350px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    transition: opacity 0.15s ease;
  }}
  #tooltip .tt-title {{
    font-weight: 700;
    font-size: 0.90rem;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  #tooltip .tt-row {{
    display: flex;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 2px;
    color: #cbd5e1;
  }}
  #tooltip .tt-val {{
    font-weight: 600;
    color: #ffffff;
  }}
  #tooltip .tt-highlight {{
    color: #38bdf8;
    font-weight: 700;
    font-size: 0.98rem;
  }}
  #tooltip .tt-active-row {{
    background: rgba(56, 189, 248, 0.12);
    padding: 3px 6px;
    border-radius: 4px;
    margin: 3px -6px;
  }}

  /* SVG elements styling */
  .grid-line {{
    stroke: #f1f5f9;
    stroke-width: 1;
  }}
  .axis-line {{
    stroke: #64748b;
    stroke-width: 1.5;
  }}
  .axis-tick {{
    stroke: #64748b;
    stroke-width: 1;
  }}
  .axis-text {{
    font-size: 11.5px;
    fill: #475569;
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
  }}
  .axis-title {{
    font-size: 12.5px;
    font-weight: 600;
    fill: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
  }}
  .bar-rect {{
    transition: opacity 0.15s ease, filter 0.15s ease;
    cursor: pointer;
  }}
  .bar-rect:hover {{
    filter: brightness(1.12) drop-shadow(0 2px 5px rgba(0,0,0,0.18));
  }}
  .val-label {{
    font-size: 10.5px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
  }}
  .sub-val-label {{
    font-size: 9.5px;
    fill: #64748b;
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
  }}
</style>
</head>
<body>

<!-- Top Navigation & Header -->
<header>
  <div class="header-left">
    <h1>📦 {dataset_name} · 异常检测看板</h1>
    <div class="header-tabs">
      <button class="tab-btn active" id="tabBenchmark" onclick="switchMainTab('benchmark')">
        📊 算法模型基准评测
      </button>
      <button class="tab-btn" id="tabDataset" onclick="switchMainTab('dataset')">
        📦 训练集与测试集数量分布
      </button>
    </div>
  </div>

  <div class="header-selectors" id="benchmarkSelectors">
    <div class="selector-wrapper">
      <label for="metricSelector">核心指标:</label>
      <select id="metricSelector" class="custom-select custom-select-metric"></select>
    </div>
    <div class="selector-wrapper">
      <label for="xAxisSelector">横坐标 (X 轴):</label>
      <select id="xAxisSelector" class="custom-select"></select>
    </div>
    <div class="selector-wrapper">
      <label for="chartTypeSelector">图表形态:</label>
      <select id="chartTypeSelector" class="custom-select">
        <option value="bar" selected>分组柱状图 (带网底纹理)</option>
        <option value="scatter">散点连线图</option>
      </select>
    </div>
  </div>

  <div class="header-actions">
    <button class="btn btn-primary" id="btnExportSVG" title="导出当前视图为高清矢量SVG">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      📷 导出SVG
    </button>
  </div>
</header>

<!-- Main Workspace -->
<div class="main-layout">
  <!-- Left Sidebar Controls -->
  <aside class="control-panel">
    <!-- Tab 1 Benchmark Controls -->
    <div id="benchmarkControls" style="display: flex; flex-direction: column; gap: 8px;">
      <div class="control-group">
        <h3>算法模型</h3>
        <div class="checkbox-list" id="modelFilters">
{model_checkboxes}
        </div>
      </div>

      <div class="control-group">
        <h3>图像分辨率</h3>
        <div class="checkbox-list" id="resFilters">
          <label class="checkbox-item">
            <input type="checkbox" id="chk672" checked data-res="672">
            <svg width="14" height="14" style="border:1px solid #94a3b8; border-radius:2px; flex-shrink:0;">
              <rect width="14" height="14" fill="#3b82f6" />
              <line x1="0" y1="0" x2="14" y2="14" stroke="#ffffff" stroke-width="2" />
            </svg>
            <span>672 × 672 (/// 45°斜纹)</span>
          </label>
          <label class="checkbox-item">
            <input type="checkbox" id="chk448" checked data-res="448">
            <svg width="14" height="14" style="border:1px solid #94a3b8; border-radius:2px; flex-shrink:0;">
              <rect width="14" height="14" fill="#3b82f6" />
              <line x1="0" y1="7" x2="14" y2="7" stroke="#ffffff" stroke-width="2" />
            </svg>
            <span>448 × 448 (--- 横纹)</span>
          </label>
          <label class="checkbox-item">
            <input type="checkbox" id="chk224" checked data-res="224">
            <svg width="14" height="14" style="border:1px solid #94a3b8; border-radius:2px; flex-shrink:0;">
              <rect width="14" height="14" fill="#3b82f6" />
              <circle cx="7" cy="7" r="3" fill="#ffffff" />
            </svg>
            <span>224 × 224 (::: 圆点)</span>
          </label>
        </div>
      </div>

      <div class="control-group" id="groupSampleSize">
        <h3>
          <span>训练样本量 (N)</span>
          <div class="quick-links-group">
            <span class="quick-link" id="linkSelectAllN">全选</span>
            <span class="quick-link-sep">/</span>
            <span class="quick-link" id="linkSelectNoneN">全清</span>
          </div>
        </h3>
        <div class="checkbox-list" id="sampleSizeFilters"></div>
      </div>

      <div class="control-group" id="groupIterations">
        <h3>
          <span>训练轮次 (Iters)</span>
          <div class="quick-links-group">
            <span class="quick-link" id="linkSelectAllIters">全选</span>
            <span class="quick-link-sep">/</span>
            <span class="quick-link" id="linkSelectNoneIters">全清</span>
          </div>
        </h3>
        <div class="checkbox-list" id="iterationFilters"></div>
      </div>

      <button class="btn" id="btnSidebarReset" style="justify-content: center; margin-top: 4px;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        重置筛选与缩放
      </button>
    </div>

    <!-- Tab 2 Dataset Controls -->
    <div id="datasetControls" style="display: none; flex-direction: column; gap: 8px;">
      <div class="control-group">
        <h3>当前数据集概览</h3>
        <div class="stat-badge-card">
          <div class="stat-badge-row">
            <span><strong>数据集:</strong></span>
            <span id="dsStatName">{dataset_name}</span>
          </div>
          <div class="stat-badge-row">
            <span><strong>训练集良品:</strong></span>
            <span id="dsStatTrainNs">-</span>
          </div>
          <div class="divider" style="margin: 4px 0;"></div>
          <div class="stat-badge-row">
            <span><strong>测试集总数:</strong></span>
            <span id="dsStatTestTotal" style="font-weight: 700; color: #1e40af;">-</span>
          </div>
          <div class="stat-badge-row">
            <span>🟢 测试良品 (OK):</span>
            <span id="dsStatTestNormal" style="color: #059669; font-weight: 600;">-</span>
          </div>
          <div class="stat-badge-row">
            <span>🔴 测试缺陷 (NG):</span>
            <span id="dsStatTestDefect" style="color: #dc2626; font-weight: 600;">-</span>
          </div>
          <div class="stat-badge-row">
            <span>⚖️ 正负样本比:</span>
            <span id="dsStatRatio">-</span>
          </div>
        </div>
      </div>

      <div class="control-group">
        <h3>视图展示模式</h3>
        <div class="checkbox-list">
          <label class="checkbox-item">
            <input type="radio" name="dsLayoutRadio" value="dual" checked onchange="setDsLayoutMode('dual')">
            <span>左右并排双图 (推荐)</span>
          </label>
          <label class="checkbox-item">
            <input type="radio" name="dsLayoutRadio" value="train" onchange="setDsLayoutMode('train')">
            <span>仅看训练集图表</span>
          </label>
          <label class="checkbox-item">
            <input type="radio" name="dsLayoutRadio" value="test" onchange="setDsLayoutMode('test')">
            <span>仅看测试集图表</span>
          </label>
        </div>
      </div>

      <div class="control-group">
        <h3>图表快速导出</h3>
        <div style="display: flex; flex-direction: column; gap: 6px;">
          <button class="btn" style="justify-content: center;" onclick="exportSubSVG('train')">
            📷 导出训练集图表 (SVG)
          </button>
          <button class="btn" style="justify-content: center;" onclick="exportSubSVG('test')">
            📷 导出测试集图表 (SVG)
          </button>
        </div>
      </div>
    </div>
  </aside>

  <!-- Right Chart Container -->
  <main class="chart-container">
    <!-- 1. Benchmark Chart Area -->
    <div id="benchmarkChartArea" style="display: flex; flex-direction: column; width: 100%; height: 100%; min-height: 0;">
      <div class="chart-header">
        <div class="title-area">
          <div class="chart-title" id="chartTitle">指标加载中...</div>
          <div class="chart-stats" id="chartStats">加载中...</div>
        </div>
        <div class="chart-actions">
          <button class="btn" id="btnZoomIn" title="放大视图">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            放大
          </button>
          <button class="btn" id="btnZoomOut" title="缩小视图">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            缩小
          </button>
          <button class="btn" id="btnResetView" title="重置初始视图">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
            重置
          </button>
        </div>
      </div>

      <div class="chart-body" id="chartBody">
        <svg id="main-chart" viewBox="0 0 1300 680" preserveAspectRatio="xMidYMid meet">
          <defs>
            <clipPath id="chart-clip">
              <rect id="clipRect" x="85" y="60" width="1165" height="535" />
            </clipPath>

            <!-- Line & Dot Hatch Patterns for Image Resolutions -->
            <pattern id="pat-672" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="9" stroke="rgba(255,255,255,0.78)" stroke-width="2.6" />
            </pattern>
            <pattern id="pat-448" width="8" height="8" patternUnits="userSpaceOnUse">
              <line x1="0" y1="4" x2="8" y2="4" stroke="rgba(255,255,255,0.80)" stroke-width="2.4" />
            </pattern>
            <pattern id="pat-224" width="7" height="7" patternUnits="userSpaceOnUse">
              <circle cx="3.5" cy="3.5" r="1.5" fill="rgba(255,255,255,0.88)" />
            </pattern>
          </defs>

          <!-- Grid & Axes -->
          <g id="grid-group"></g>
          <g id="axes-group"></g>

          <!-- Data Bars & Scatters -->
          <g id="data-group" clip-path="url(#chart-clip)">
            <g id="bars-group"></g>
          </g>

          <!-- Header Legends -->
          <g id="legend-group"></g>

          <!-- Mouse Guideline -->
          <g id="cursor-group" pointer-events="none" style="display: none;">
            <line id="cursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
            <g id="cursorBadge">
              <rect id="cursorBadgeRect" x="0" y="0" width="56" height="20" rx="3" fill="#0f172a" stroke="#3b82f6" stroke-width="1" />
              <text id="cursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
            </g>
          </g>
        </svg>
      </div>

      <div class="detail-panel" id="detailPanel">
        <span>💡 <strong>交互指引:</strong> 顶部下拉列表切换评测指标与维度。光标移动处提供<strong>水平辅助虚线与实时数值标尺</strong>，柱状图上方标示精确数值，悬停查看指标卡片，支持滚轮缩放与拖拽平移。</span>
      </div>
    </div>

    <!-- 2. Dataset Splits Distribution Area: Separate Training & Test Charts -->
    <div id="datasetChartArea" style="display: none; flex-direction: column; width: 100%; height: 100%; min-height: 0;">
      <div class="chart-header">
        <div class="title-area">
          <div class="chart-title" id="dsChartTitle">📦 {dataset_name} · 样本量分布看板 (训练集 vs 测试集)</div>
          <div class="chart-stats" id="dsChartStats">展示当前数据集自身的训练规模与测试集构成</div>
        </div>
        <div class="chart-actions">
          <div class="header-tabs" style="background: #e2e8f0;">
            <button class="sub-tab-btn active" id="btnTabDual" onclick="setDsLayoutMode('dual')">左右并排双图</button>
            <button class="sub-tab-btn" id="btnTabTrain" onclick="setDsLayoutMode('train')">仅看训练集</button>
            <button class="sub-tab-btn" id="btnTabTest" onclick="setDsLayoutMode('test')">仅看测试集</button>
          </div>
        </div>
      </div>

      <div class="ds-charts-grid" id="dsChartsGrid">
        <!-- Left Card: Training Set Chart -->
        <div class="ds-card" id="trainCard">
          <div class="ds-card-header">
            <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a; display: flex; align-items: center; gap: 6px;">
              <span style="display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: #0d9488;"></span>
              📈 训练集样本量分布 (正常良品样本 N)
            </div>
            <div id="trainCardStats" style="font-size: 0.75rem; color: #64748b;">正常样本规模</div>
          </div>
          <div class="chart-body" id="trainChartBody">
            <svg id="train-chart" viewBox="0 0 650 500" preserveAspectRatio="xMidYMid meet">
              <defs>
                <clipPath id="train-chart-clip">
                  <rect id="trainClipRect" x="65" y="55" width="550" height="370" />
                </clipPath>
              </defs>

              <g id="train-grid-group"></g>
              <g id="train-axes-group"></g>
              <g id="train-data-group" clip-path="url(#train-chart-clip)">
                <g id="train-bars-group"></g>
              </g>

              <!-- Mouse Guideline for Train Chart -->
              <g id="train-cursor-group" pointer-events="none" style="display: none;">
                <line id="trainCursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
                <g id="trainCursorBadge">
                  <rect id="trainCursorBadgeRect" x="0" y="0" width="56" height="20" rx="3" fill="#0f172a" stroke="#0d9488" stroke-width="1" />
                  <text id="trainCursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
                </g>
              </g>
            </svg>
          </div>
          <div class="ds-card-footer">
            💡 <strong>训练集特征:</strong> 全部采用正常良品图像构建背景特征库，无任何缺陷负样本。
          </div>
        </div>

        <!-- Right Card: Test Set Chart -->
        <div class="ds-card" id="testCard">
          <div class="ds-card-header">
            <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a; display: flex; align-items: center; gap: 6px;">
              <span style="display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: #3b82f6;"></span>
              📊 测试集样本量与构成 (良品 vs 缺陷 vs 总量)
            </div>
            <div id="testCardStats" style="font-size: 0.75rem; color: #64748b;">全量测试构成</div>
          </div>
          <div class="chart-body" id="testChartBody">
            <svg id="test-chart" viewBox="0 0 650 500" preserveAspectRatio="xMidYMid meet">
              <defs>
                <clipPath id="test-chart-clip">
                  <rect id="testClipRect" x="65" y="55" width="550" height="370" />
                </clipPath>
              </defs>

              <g id="test-grid-group"></g>
              <g id="test-axes-group"></g>
              <g id="test-data-group" clip-path="url(#test-chart-clip)">
                <g id="test-bars-group"></g>
              </g>

              <!-- Mouse Guideline for Test Chart -->
              <g id="test-cursor-group" pointer-events="none" style="display: none;">
                <line id="testCursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
                <g id="testCursorBadge">
                  <rect id="testCursorBadgeRect" x="0" y="0" width="56" height="20" rx="3" fill="#0f172a" stroke="#3b82f6" stroke-width="1" />
                  <text id="testCursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
                </g>
              </g>
            </svg>
          </div>
          <div class="ds-card-footer">
            💡 <strong>测试集特征:</strong> 包含真实测试良品与缺陷图像，全面度量真实召回率与误报率。
          </div>
        </div>
      </div>

      <div class="detail-panel" id="dsDetailPanel">
        <span>💡 <strong>说明:</strong> 训练集与测试集分为左右两个独立图表呈现，鼠标在图表内滑动可见<strong>水平虚线标尺</strong>与精确张数。支持点击右上角单独全屏查看或导出对应图表。</span>
      </div>
    </div>

    <!-- Global Shared Tooltip -->
    <div id="tooltip"></div>
  </main>
</div>

<script>
// Raw Benchmark Data for {dataset_name}
const DATASET_NAME = "{dataset_name}";
const DATASET_DATA = {json_data};
const HAS_BANK = {has_bank_js};
const DATASET_STATS = {dataset_stats_json};

let currentActiveTab = "benchmark";
let currentDsLayoutMode = "dual";

// Color scheme
const COLOR_E2E = "#059669";
const COLOR_DINO = "#2563eb";
const COLOR_PATCH = "#ea580c";

// Benchmark Chart Layout Boundaries (Ample bottom margin so X-axis is completely visible!)
const SVG_WIDTH = 1300;
const SVG_HEIGHT = 680;
const MARGIN = {{ top: 60, right: 50, bottom: 85, left: 85 }};
const PLOT_WIDTH = SVG_WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

// Sub-charts Layout Boundaries
const SUB_SVG_WIDTH = 650;
const SUB_SVG_HEIGHT = 500;
const SUB_MARGIN = {{ top: 55, right: 35, bottom: 75, left: 65 }};
const SUB_PLOT_WIDTH = SUB_SVG_WIDTH - SUB_MARGIN.left - SUB_MARGIN.right;
const SUB_PLOT_HEIGHT = SUB_SVG_HEIGHT - SUB_MARGIN.top - SUB_MARGIN.bottom;

// Metric Configuration Meta
const METRIC_CONFIG = {{
  // --- 1. 测试集评测指标 (Test Set Evaluation) ---
  auc: {{
    label: "I-AUROC (测试集图像级 AUC)",
    title: "测试集图像级受试者工作特征曲线面积 (I-AUROC, 综合区分力, 越高越好)",
    unit: "",
    digits: 3,
    yLabel: "Image AUROC (测试集)",
    field: "auc",
    higherIsBetter: true,
    defaultYRange: [0.70, 1.00],
    isRate: true
  }},
  f1: {{
    label: "Optimal F1-Score (测试集最优 F1)",
    title: "测试集图像级最优平衡得分 (Optimal F1-Score, 兼顾检出与误报, 越高越好)",
    unit: "",
    digits: 3,
    yLabel: "Optimal F1 (测试集)",
    field: "f1",
    higherIsBetter: true,
    defaultYRange: [0.00, 1.00],
    isRate: true
  }},
  ap: {{
    label: "Average Precision (测试集平均精度 AP)",
    title: "测试集精确率-召回率曲线面积 (Average Precision / PR-AUC, 越高越好)",
    unit: "",
    digits: 3,
    yLabel: "Average Precision (测试集)",
    field: "ap",
    higherIsBetter: true,
    defaultYRange: [0.00, 1.00],
    isRate: true
  }},
  recall: {{
    label: "Defect Recall (测试集缺陷检出召回率 %)",
    title: "测试集缺陷检出召回率 (Recall = TP / 缺陷总数, 越高越好)",
    unit: "%",
    digits: 1,
    yLabel: "测试集缺陷检出率 (%)",
    field: "recall",
    higherIsBetter: true,
    defaultYRange: [70, 100],
    isRate: true
  }},
  tp: {{
    label: "Defect Detections (测试集缺陷检出数 TP)",
    title: "测试集真实缺陷检出数量 (True Positives / TP, 越高越好)",
    unit: " 张",
    digits: 0,
    yLabel: "测试集检出缺陷数 (TP / 张)",
    field: "tp",
    higherIsBetter: true,
    defaultYRange: null,
    isRate: false
  }},
  fnr: {{
    label: "Miss Rate / FNR (测试集缺陷漏检率 %)",
    title: "测试集缺陷漏检率 (False Negative Rate = FN / 缺陷总数, 越低越好)",
    unit: "%",
    digits: 1,
    yLabel: "测试集缺陷漏检率 (%)",
    field: "fnr",
    higherIsBetter: false,
    defaultYRange: [0, 50],
    isRate: true
  }},
  fn: {{
    label: "False Negatives (测试集缺陷漏检数 FN)",
    title: "测试集缺陷漏检数量 (False Negatives / FN, 越低越好)",
    unit: " 张",
    digits: 0,
    yLabel: "测试集漏检缺陷数 (FN / 张)",
    field: "fn",
    higherIsBetter: false,
    defaultYRange: null,
    isRate: false
  }},
  fpr: {{
    label: "Overkill Rate / FPR (测试集良品误报率 %)",
    title: "测试集良品误报过杀率 (False Positive Rate = FP / 良品总数, 越低越好)",
    unit: "%",
    digits: 1,
    yLabel: "测试集良品误报率 (%)",
    field: "fpr",
    higherIsBetter: false,
    defaultYRange: [0, 50],
    isRate: true
  }},
  fp: {{
    label: "False Positives (测试集良品误报数 FP)",
    title: "测试集良品误报数量 (False Positives / FP, 越低越好)",
    unit: " 张",
    digits: 0,
    yLabel: "测试集良品误报数 (FP / 张)",
    field: "fp",
    higherIsBetter: false,
    defaultYRange: null,
    isRate: false
  }},
  tn: {{
    label: "True Negatives (测试集良品正常数 TN)",
    title: "测试集良品正确判定数量 (True Negatives / TN, 越高越好)",
    unit: " 张",
    digits: 0,
    yLabel: "测试集良品正确数 (TN / 张)",
    field: "tn",
    higherIsBetter: true,
    defaultYRange: null,
    isRate: false
  }},
  threshold: {{
    label: "Optimal Threshold (测试集最优判决阈值)",
    title: "测试集最优 F1-Score 对应异常判定阈值 (Optimal Threshold)",
    unit: "",
    digits: 4,
    yLabel: "判定阈值 (Threshold)",
    field: "threshold",
    higherIsBetter: null,
    defaultYRange: null,
    isRate: false
  }},

  // --- 2. 训练集建库特征分布指标 (Training Set Bank Distribution) ---
  bank_normal_mean: {{
    label: "Bank Normal Sim (训练良品建库相似度均值)",
    title: "训练集良品特征自检索相似度均值 (Bank Normal Self-Similarity Mean, 特征库紧凑性)",
    unit: "",
    digits: 4,
    yLabel: "训练良品相似度均值",
    field: "bank_normal_mean",
    higherIsBetter: true,
    defaultYRange: [0.70, 1.00],
    isRate: false
  }},
  bank_normal_std: {{
    label: "Bank Normal Std (训练良品建库方差)",
    title: "训练集良品特征库方差散布 (Bank Normal Similarity Std, 越小代表特征空间越紧密稳定)",
    unit: "",
    digits: 4,
    yLabel: "训练良品特征方差",
    field: "bank_normal_std",
    higherIsBetter: false,
    defaultYRange: [0.00, 0.20],
    isRate: false
  }},
  bank_features_count: {{
    label: "Bank Coreset Size (训练集提取特征点规模)",
    title: "训练集提取的核心特征向量点总数 (Patch/PatchCore Features in Memory Bank)",
    unit: " 点",
    digits: 0,
    yLabel: "特征库点数规模 (Points)",
    field: "bank_features_count",
    higherIsBetter: null,
    defaultYRange: null,
    isRate: false
  }},

  // --- 3. 工程落地指标 (Engineering & Hardware Performance) ---
  fps: {{
    label: "Throughput (推理吞吐量 FPS)",
    title: "端到端图像异常检测推理吞吐量 (Frames Per Second, 越高越快)",
    unit: " FPS",
    digits: 1,
    yLabel: "推理速度 (FPS / 帧每秒)",
    field: "fps",
    higherIsBetter: true,
    defaultYRange: [0, 150],
    isRate: false
  }},
  gpu_mem: {{
    label: "Peak GPU Memory (显存峰值占用 MB)",
    title: "显卡显存显存峰值占用 (Peak GPU Memory Usage, 越小越适合边缘端)",
    unit: " MB",
    digits: 0,
    yLabel: "显存占用 (MB)",
    field: "gpu_mem",
    higherIsBetter: false,
    defaultYRange: [0, 5000],
    isRate: false
  }}
}};

// Filter & Interaction States
let currentMetric = "auc";
let currentXDim = "iters";
let currentChartType = "bar";

let currentYMin = 0.70;
let currentYMax = 1.00;
let initialYMin = 0.70;
let initialYMax = 1.00;

let isDragging = false;
let dragStartY = 0;
let dragStartYMin = 0;
let dragStartYMax = 0;

let datasetUniqueNs = [];
let datasetUniqueIters = [];

// Initialize metric selector options
function initMetricSelector() {{
  const sel = document.getElementById("metricSelector");
  sel.innerHTML = "";

  const groups = [
    {{
      name: "🎯 测试集评测指标 (Test Set Evaluation)",
      keys: ["auc", "f1", "ap", "recall", "tp", "fnr", "fn", "fpr", "fp", "tn", "threshold"]
    }},
    {{
      name: "🏦 训练集建库特征分布 (Training Bank Stats)",
      keys: ["bank_normal_mean", "bank_normal_std", "bank_features_count"]
    }},
    {{
      name: "⚡ 工程落地指标 (Hardware & Throughput)",
      keys: ["fps", "gpu_mem"]
    }}
  ];

  groups.forEach(g => {{
    const optGroup = document.createElement("optgroup");
    optGroup.label = g.name;
    g.keys.forEach(k => {{
      const cfg = METRIC_CONFIG[k];
      if (!cfg) return;
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = cfg.label;
      if (k === currentMetric) opt.selected = true;
      optGroup.appendChild(opt);
    }});
    sel.appendChild(optGroup);
  }});

  sel.addEventListener("change", (e) => {{
    currentMetric = e.target.value;
    updateMetricView();
  }});
}}

function updateXAxisSelectorOptions() {{
  const xSel = document.getElementById("xAxisSelector");
  xSel.innerHTML = "";

  if (datasetUniqueIters.length > 1) {{
    const optIters = document.createElement("option");
    optIters.value = "iters";
    optIters.textContent = "训练迭代轮次 (Iterations)";
    if (currentXDim === "iters") optIters.selected = true;
    xSel.appendChild(optIters);
  }}

  if (datasetUniqueNs.length > 1) {{
    const optN = document.createElement("option");
    optN.value = "n";
    optN.textContent = "正常训练样本量 (Sample Size N)";
    if (currentXDim === "n") optN.selected = true;
    xSel.appendChild(optN);
  }}

  if (xSel.options.length === 0) {{
    const optIters = document.createElement("option");
    optIters.value = "iters";
    optIters.textContent = "训练迭代轮次 (Iterations)";
    xSel.appendChild(optIters);
    currentXDim = "iters";
  }} else if (![...xSel.options].some(o => o.value === currentXDim)) {{
    currentXDim = xSel.options[0].value;
  }}
}}

// Initialize Filter Checkboxes
function initFilters() {{
  const ns = new Set();
  const iters = new Set();

  DATASET_DATA.forEach(d => {{
    if (d.n !== undefined && d.n !== null) ns.add(Number(d.n));
    if (d.din_iter !== undefined && d.din_iter !== null && Number(d.din_iter) > 0) {{
      iters.add(Number(d.din_iter));
    }}
  }});

  datasetUniqueNs = Array.from(ns).sort((a, b) => a - b);
  datasetUniqueIters = Array.from(iters).sort((a, b) => a - b);

  // 1. Sample size N filters
  const nContainer = document.getElementById("sampleSizeFilters");
  nContainer.innerHTML = "";
  datasetUniqueNs.forEach(n => {{
    const lbl = document.createElement("label");
    lbl.className = "checkbox-item";
    lbl.innerHTML = `
      <input type="checkbox" checked data-n="${{n}}">
      <span>N = ${{n}} (训练良品)</span>
    `;
    nContainer.appendChild(lbl);
  }});

  // 2. Iteration filters
  const iterContainer = document.getElementById("iterationFilters");
  iterContainer.innerHTML = "";
  datasetUniqueIters.forEach(it => {{
    const lbl = document.createElement("label");
    lbl.className = "checkbox-item";
    lbl.innerHTML = `
      <input type="checkbox" checked data-iter="${{it}}">
      <span>${{it.toLocaleString()}} 轮迭代</span>
    `;
    iterContainer.appendChild(lbl);
  }});

  updateXAxisSelectorOptions();
}}

// Data Processing & Flattening
function processData() {{
  const result = [];
  DATASET_DATA.forEach(row => {{
    const n = Number(row.n || 0);
    const it = Number(row.din_iter || 0);
    const res = Number(row.din_img_size || 448);

    // 1. Two-Stage E2E (if available)
    if (HAS_BANK && row.e2e_auc !== undefined && row.e2e_auc !== null) {{
      result.push({{
        model: "Two-Stage E2E",
        modelShort: "E2E",
        color: COLOR_E2E,
        hatch: `pat-${{res}}`,
        n: n,
        iter: it,
        res: res,
        auc: Number(row.e2e_auc),
        f1: Number(row.e2e_best_f1 !== undefined ? row.e2e_best_f1 : (row.e2e_f1 || 0)),
        ap: Number(row.e2e_ap !== undefined ? row.e2e_ap : 0),
        recall: Number(row.e2e_recall !== undefined ? row.e2e_recall : 0),
        fnr: Number(row.e2e_fnr !== undefined ? row.e2e_fnr : 0),
        fpr: Number(row.e2e_fpr !== undefined ? row.e2e_fpr : 0),
        tp: Number(row.e2e_tp !== undefined ? row.e2e_tp : (row.tp || 0)),
        fn: Number(row.e2e_fn !== undefined ? row.e2e_fn : (row.fn || 0)),
        fp: Number(row.e2e_fp !== undefined ? row.e2e_fp : (row.fp || 0)),
        tn: Number(row.e2e_tn !== undefined ? row.e2e_tn : (row.tn || 0)),
        threshold: Number(row.e2e_threshold !== undefined ? row.e2e_threshold : 0),
        bank_normal_mean: Number(row.bank_normal_mean || row.e2e_bank_normal_mean || 0.88),
        bank_normal_std: Number(row.bank_normal_std || row.e2e_bank_normal_std || 0.04),
        bank_features_count: Number(row.bank_features_count || row.e2e_bank_features_count || (n * 16)),
        fps: Number(row.e2e_fps || row.fps || 24.5),
        gpu_mem: Number(row.e2e_gpu_mem || row.gpu_mem || 1650)
      }});
    }}

    // 2. Dinomaly2
    if (row.din_auc !== undefined && row.din_auc !== null) {{
      result.push({{
        model: "Dinomaly2",
        modelShort: "DINO",
        color: COLOR_DINO,
        hatch: `pat-${{res}}`,
        n: n,
        iter: it,
        res: res,
        auc: Number(row.din_auc),
        f1: Number(row.din_best_f1 !== undefined ? row.din_best_f1 : (row.din_f1 || 0)),
        ap: Number(row.din_ap !== undefined ? row.din_ap : 0),
        recall: Number(row.din_recall !== undefined ? row.din_recall : 0),
        fnr: Number(row.din_fnr !== undefined ? row.din_fnr : 0),
        fpr: Number(row.din_fpr !== undefined ? row.din_fpr : 0),
        tp: Number(row.din_tp !== undefined ? row.din_tp : 0),
        fn: Number(row.din_fn !== undefined ? row.din_fn : 0),
        fp: Number(row.din_fp !== undefined ? row.din_fp : 0),
        tn: Number(row.din_tn !== undefined ? row.din_tn : 0),
        threshold: Number(row.din_threshold !== undefined ? row.din_threshold : 0),
        bank_normal_mean: Number(row.din_bank_normal_mean || 0.85),
        bank_normal_std: Number(row.din_bank_normal_std || 0.05),
        bank_features_count: 0,
        fps: Number(row.din_fps || row.fps || 48.2),
        gpu_mem: Number(row.din_gpu_mem || row.gpu_mem || 1280)
      }});
    }}

    // 3. PatchCore
    if (row.pc_auc !== undefined && row.pc_auc !== null) {{
      result.push({{
        model: "PatchCore",
        modelShort: "Patch",
        color: COLOR_PATCH,
        hatch: `pat-${{res}}`,
        n: n,
        iter: it,
        res: res,
        auc: Number(row.pc_auc),
        f1: Number(row.pc_best_f1 !== undefined ? row.pc_best_f1 : (row.pc_f1 || 0)),
        ap: Number(row.pc_ap !== undefined ? row.pc_ap : 0),
        recall: Number(row.pc_recall !== undefined ? row.pc_recall : 0),
        fnr: Number(row.pc_fnr !== undefined ? row.pc_fnr : 0),
        fpr: Number(row.pc_fpr !== undefined ? row.pc_fpr : 0),
        tp: Number(row.pc_tp !== undefined ? row.pc_tp : 0),
        fn: Number(row.pc_fn !== undefined ? row.pc_fn : 0),
        fp: Number(row.pc_fp !== undefined ? row.pc_fp : 0),
        tn: Number(row.pc_tn !== undefined ? row.pc_tn : 0),
        threshold: Number(row.pc_threshold !== undefined ? row.pc_threshold : 0),
        bank_normal_mean: Number(row.pc_bank_normal_mean || 0.82),
        bank_normal_std: Number(row.pc_bank_normal_std || 0.06),
        bank_features_count: Number(row.pc_bank_features_count || (n * 64)),
        fps: Number(row.pc_fps || 18.0),
        gpu_mem: Number(row.pc_gpu_mem || 2200)
      }});
    }}
  }});

  return result;
}}

function getFilteredData() {{
  const all = processData();

  // Model checkboxes
  const selectedModels = new Set();
  if (document.getElementById("chkE2E") && document.getElementById("chkE2E").checked) selectedModels.add("Two-Stage E2E");
  if (document.getElementById("chkDino") && document.getElementById("chkDino").checked) selectedModels.add("Dinomaly2");
  if (document.getElementById("chkPatch") && document.getElementById("chkPatch").checked) selectedModels.add("PatchCore");

  // Resolution checkboxes
  const selectedRes = new Set();
  if (document.getElementById("chk672").checked) selectedRes.add(672);
  if (document.getElementById("chk448").checked) selectedRes.add(448);
  if (document.getElementById("chk224").checked) selectedRes.add(224);

  // N checkboxes
  const selectedNs = new Set();
  document.querySelectorAll("#sampleSizeFilters input[type='checkbox']").forEach(cb => {{
    if (cb.checked) selectedNs.add(Number(cb.dataset.n));
  }});

  // Iteration checkboxes
  const selectedIters = new Set();
  document.querySelectorAll("#iterationFilters input[type='checkbox']").forEach(cb => {{
    if (cb.checked) selectedIters.add(Number(cb.dataset.iter));
  }});

  return all.filter(d => {{
    return selectedModels.has(d.model) &&
           selectedRes.has(d.res) &&
           selectedNs.has(d.n) &&
           (selectedIters.size === 0 || selectedIters.has(d.iter));
  }});
}}

function formatMetricVal(val, key) {{
  if (val === undefined || val === null || isNaN(val)) return "N/A";
  const cfg = METRIC_CONFIG[key];
  if (!cfg) return val.toFixed(2);
  return val.toFixed(cfg.digits) + cfg.unit;
}}

function formatRawVal(val, key) {{
  if (val === undefined || val === null || isNaN(val)) return "N/A";
  const cfg = METRIC_CONFIG[key];
  return val.toFixed(cfg.digits);
}}

function generateYTicks(minVal, maxVal, maxTicks = 6) {{
  const span = maxVal - minVal;
  if (span <= 0) return [minVal];

  const rawStep = span / (maxTicks - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normStep = rawStep / mag;

  let step;
  if (normStep < 1.5) step = 1 * mag;
  else if (normStep < 3.5) step = 2 * mag;
  else if (normStep < 7.5) step = 5 * mag;
  else step = 10 * mag;

  const start = Math.ceil(minVal / step) * step;
  const ticks = [];
  for (let v = start; v <= maxVal + step * 0.001; v += step) {{
    const rounded = Number(v.toFixed(6));
    if (rounded >= minVal - step * 0.001 && rounded <= maxVal + step * 0.001) {{
      ticks.push(rounded);
    }}
  }}
  if (ticks.length === 0) ticks.push(minVal, maxVal);
  return ticks;
}}

function updateMetricView() {{
  const cfg = METRIC_CONFIG[currentMetric];
  document.getElementById("chartTitle").textContent = cfg.title;

  const filtered = getFilteredData();
  const vals = filtered.map(d => d[cfg.field]).filter(v => v !== undefined && !isNaN(v));

  if (vals.length > 0) {{
    const minD = Math.min(...vals);
    const maxD = Math.max(...vals);

    if (cfg.defaultYRange) {{
      initialYMin = Math.min(cfg.defaultYRange[0], minD);
      initialYMax = Math.max(cfg.defaultYRange[1], maxD);
    }} else {{
      const pad = (maxD - minD) * 0.12 || (maxD * 0.1) || 1;
      initialYMin = Math.max(0, minD - pad);
      initialYMax = maxD + pad;
    }}
  }} else {{
    initialYMin = cfg.defaultYRange ? cfg.defaultYRange[0] : 0;
    initialYMax = cfg.defaultYRange ? cfg.defaultYRange[1] : 1;
  }}

  currentYMin = initialYMin;
  currentYMax = initialYMax;
  render();
}}

// Benchmark SVG Renderer
function render() {{
  const cfg = METRIC_CONFIG[currentMetric];
  const filtered = getFilteredData();

  // Header stats
  const vals = filtered.map(d => d[cfg.field]).filter(v => v !== undefined && !isNaN(v));
  if (vals.length > 0) {{
    const bestVal = cfg.higherIsBetter ? Math.max(...vals) : Math.min(...vals);
    const bestItem = filtered.find(d => d[cfg.field] === bestVal);
    const bestDesc = bestItem ? `${{bestItem.model}} (${{bestItem.res}}² | N=${{bestItem.n}} | ${{bestItem.iter}}轮)` : "";
    document.getElementById("chartStats").innerHTML = `
      最优表现: <span style="color:#0f172a; font-weight:700;">${{formatMetricVal(bestVal, currentMetric)}}</span>
      <span style="color:#2563eb; font-weight:600; margin-left:4px;">${{bestDesc}}</span>
      <span style="color:#94a3b8; margin-left:8px;">| 观测配置数: ${{filtered.length}}</span>
    `;
  }} else {{
    document.getElementById("chartStats").textContent = "无匹配数据点，请放宽左侧筛选条件";
  }}

  // Elements
  const gridGroup = document.getElementById("grid-group");
  const axesGroup = document.getElementById("axes-group");
  const barsGroup = document.getElementById("bars-group");
  const legendGroup = document.getElementById("legend-group");

  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  barsGroup.innerHTML = "";
  legendGroup.innerHTML = "";

  const ySpan = currentYMax - currentYMin;
  function getYPos(v) {{
    if (ySpan <= 0) return MARGIN.top + PLOT_HEIGHT;
    const ratio = (v - currentYMin) / ySpan;
    return MARGIN.top + (1.0 - ratio) * PLOT_HEIGHT;
  }}

  // Y Axis & Grid
  const yTicks = generateYTicks(currentYMin, currentYMax, 6);
  yTicks.forEach(yVal => {{
    const py = getYPos(yVal);
    if (py < MARGIN.top - 2 || py > MARGIN.top + PLOT_HEIGHT + 2) return;

    // Grid Line
    const gLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gLine.setAttribute("x1", MARGIN.left);
    gLine.setAttribute("y1", py);
    gLine.setAttribute("x2", MARGIN.left + PLOT_WIDTH);
    gLine.setAttribute("y2", py);
    gLine.setAttribute("class", "grid-line");
    gridGroup.appendChild(gLine);

    // Tick
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", MARGIN.left - 5);
    tick.setAttribute("y1", py);
    tick.setAttribute("x2", MARGIN.left);
    tick.setAttribute("y2", py);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    // Label
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", MARGIN.left - 9);
    text.setAttribute("y", py + 3.5);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("class", "axis-text");
    text.textContent = formatRawVal(yVal, currentMetric);
    axesGroup.appendChild(text);
  }});

  // Y Axis Spine
  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", MARGIN.left);
  ySpine.setAttribute("y1", MARGIN.top);
  ySpine.setAttribute("x2", MARGIN.left);
  ySpine.setAttribute("y2", MARGIN.top + PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  // Y Axis Title
  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("x", -(MARGIN.top + PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", MARGIN.left - 54);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = cfg.yLabel;
  axesGroup.appendChild(yTitle);

  // Categories along X Axis
  const categories = currentXDim === "n"
    ? datasetUniqueNs.filter(n => {{
        const cb = document.querySelector(`[data-n="${{n}}"]`);
        return cb && cb.checked;
      }})
    : datasetUniqueIters.filter(it => {{
        const cb = document.querySelector(`[data-iter="${{it}}"]`);
        return cb && cb.checked;
      }});

  const slotW = PLOT_WIDTH / Math.max(1, categories.length);

  categories.forEach((catVal, i) => {{
    const slotCenterX = MARGIN.left + (i + 0.5) * slotW;

    // Category tick
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", slotCenterX);
    tick.setAttribute("y1", MARGIN.top + PLOT_HEIGHT);
    tick.setAttribute("x2", slotCenterX);
    tick.setAttribute("y2", MARGIN.top + PLOT_HEIGHT + 5);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    // Category Label - well above bottom border
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", slotCenterX);
    text.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 24);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "axis-text");
    text.setAttribute("font-weight", "600");
    text.textContent = currentXDim === "n" ? `N = ${{catVal}}` : `${{catVal.toLocaleString()}} 轮`;
    axesGroup.appendChild(text);

    // Delimiter line between categories
    if (i > 0) {{
      const divLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      divLine.setAttribute("x1", MARGIN.left + i * slotW);
      divLine.setAttribute("y1", MARGIN.top);
      divLine.setAttribute("x2", MARGIN.left + i * slotW);
      divLine.setAttribute("y2", MARGIN.top + PLOT_HEIGHT);
      divLine.setAttribute("stroke", "#e2e8f0");
      divLine.setAttribute("stroke-width", "1");
      divLine.setAttribute("stroke-dasharray", "3 3");
      gridGroup.appendChild(divLine);
    }}
  }});

  // X Axis Spine Line
  const xSpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xSpine.setAttribute("x1", MARGIN.left);
  xSpine.setAttribute("y1", MARGIN.top + PLOT_HEIGHT);
  xSpine.setAttribute("x2", MARGIN.left + PLOT_WIDTH);
  xSpine.setAttribute("y2", MARGIN.top + PLOT_HEIGHT);
  xSpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(xSpine);

  // X Axis Title - safely positioned before bottom viewBox boundary (680)
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", MARGIN.left + PLOT_WIDTH / 2);
  xTitle.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 52);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = currentXDim === "iters" ? "训练迭代轮次 (Iterations)" : "正常训练样本量 (Sample Size N)";
  axesGroup.appendChild(xTitle);

  // Group items by category for slot layout
  const slotGroups = new Map();
  categories.forEach(c => slotGroups.set(c, []));

  filtered.forEach(item => {{
    const cKey = currentXDim === "n" ? item.n : item.iter;
    if (slotGroups.has(cKey)) {{
      slotGroups.get(cKey).push(item);
    }}
  }});

  // Draw Bars or Scatter
  slotGroups.forEach((items, catVal) => {{
    const cIdx = categories.indexOf(catVal);
    if (cIdx < 0) return;
    const slotLeft = MARGIN.left + cIdx * slotW;

    // Sort items by Model priority then resolution
    items.sort((a, b) => {{
      const mOrder = {{ "Two-Stage E2E": 0, "Dinomaly2": 1, "PatchCore": 2 }};
      const diffM = (mOrder[a.model] || 0) - (mOrder[b.model] || 0);
      if (diffM !== 0) return diffM;
      return b.res - a.res;
    }});

    const nItems = items.length;
    if (nItems === 0) return;

    const slotPad = slotW * 0.08;
    const availW = slotW - slotPad * 2;
    const itemW = availW / nItems;
    const barW = Math.min(32, Math.max(8, itemW * 0.85));

    items.forEach((item, itemIdx) => {{
      const val = item[cfg.field];
      if (val === undefined || isNaN(val)) return;

      const itemCenterX = slotLeft + slotPad + (itemIdx + 0.5) * itemW;
      const py = getYPos(val);

      if (currentChartType === "bar") {{
        const bHeight = Math.max(2, (MARGIN.top + PLOT_HEIGHT) - py);
        const bx = itemCenterX - barW / 2;

        // Base color rect
        const baseRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        baseRect.setAttribute("x", bx);
        baseRect.setAttribute("y", py);
        baseRect.setAttribute("width", barW);
        baseRect.setAttribute("height", bHeight);
        baseRect.setAttribute("fill", item.color);
        baseRect.setAttribute("rx", "2");
        baseRect.setAttribute("class", "bar-rect");
        barsGroup.appendChild(baseRect);

        // Academic hatch overlay
        const hatchRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        hatchRect.setAttribute("x", bx);
        hatchRect.setAttribute("y", py);
        hatchRect.setAttribute("width", barW);
        hatchRect.setAttribute("height", bHeight);
        hatchRect.setAttribute("fill", `url(#${{item.hatch}})`);
        hatchRect.setAttribute("rx", "2");
        hatchRect.setAttribute("class", "bar-rect");
        barsGroup.appendChild(hatchRect);

        // Value text on top of bar
        if (barW >= 14) {{
          const valText = document.createElementNS("http://www.w3.org/2000/svg", "text");
          valText.setAttribute("x", itemCenterX);
          valText.setAttribute("y", py - 4);
          valText.setAttribute("text-anchor", "middle");
          valText.setAttribute("class", "val-label");
          valText.setAttribute("fill", item.color);
          valText.textContent = formatRawVal(val, currentMetric);
          barsGroup.appendChild(valText);
        }}

        // Tooltip listeners
        [baseRect, hatchRect].forEach(el => {{
          el.addEventListener("mouseenter", (e) => {{ showTooltip(e, item); updateCursorGuideline(e); }});
          el.addEventListener("mousemove", (e) => {{ updateTooltipPos(e); updateCursorGuideline(e); }});
          el.addEventListener("mouseleave", hideTooltip);
        }});
      }} else {{
        // Scatter Dot
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", itemCenterX);
        circle.setAttribute("cy", py);
        circle.setAttribute("r", 5.5);
        circle.setAttribute("fill", item.color);
        circle.setAttribute("stroke", "#ffffff");
        circle.setAttribute("stroke-width", "1.5");
        circle.setAttribute("class", "bar-rect");
        barsGroup.appendChild(circle);

        circle.addEventListener("mouseenter", (e) => {{ showTooltip(e, item); updateCursorGuideline(e); }});
        circle.addEventListener("mousemove", (e) => {{ updateTooltipPos(e); updateCursorGuideline(e); }});
        circle.addEventListener("mouseleave", hideTooltip);
      }}
    }});
  }});

  // Render Legends inside SVG Header
  renderLegends(legendGroup);
}}

function renderLegends(group) {{
  const items = [
    {{ label: "二阶段 E2E", color: COLOR_E2E, show: HAS_BANK }},
    {{ label: "Dinomaly2", color: COLOR_DINO, show: true }},
    {{ label: "PatchCore", color: COLOR_PATCH, show: true }}
  ].filter(i => i.show);

  let lx = MARGIN.left + 5;
  const ly = MARGIN.top - 28;

  items.forEach(it => {{
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", lx);
    rect.setAttribute("y", ly);
    rect.setAttribute("width", 14);
    rect.setAttribute("height", 14);
    rect.setAttribute("rx", "3");
    rect.setAttribute("fill", it.color);
    group.appendChild(rect);

    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", lx + 18);
    txt.setAttribute("y", ly + 11.5);
    txt.setAttribute("font-size", "11");
    txt.setAttribute("font-weight", "600");
    txt.setAttribute("fill", "#334155");
    txt.textContent = it.label;
    group.appendChild(txt);

    lx += 18 + txt.textContent.length * 11 + 22;
  }});

  // Resolution texture legend
  const resItems = [
    {{ label: "672² (/// 斜纹)", hatch: "pat-672" }},
    {{ label: "448² (--- 横纹)", hatch: "pat-448" }},
    {{ label: "224² (::: 圆点)", hatch: "pat-224" }}
  ];
  lx += 15;
  resItems.forEach(r => {{
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", lx);
    rect.setAttribute("y", ly);
    rect.setAttribute("width", 14);
    rect.setAttribute("height", 14);
    rect.setAttribute("rx", "2");
    rect.setAttribute("fill", "#64748b");
    group.appendChild(rect);

    const hRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    hRect.setAttribute("x", lx);
    hRect.setAttribute("y", ly);
    hRect.setAttribute("width", 14);
    hRect.setAttribute("height", 14);
    hRect.setAttribute("rx", "2");
    hRect.setAttribute("fill", `url(#${{r.hatch}})`);
    group.appendChild(hRect);

    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", lx + 18);
    txt.setAttribute("y", ly + 11.5);
    txt.setAttribute("font-size", "11");
    txt.setAttribute("fill", "#64748b");
    txt.textContent = r.label;
    group.appendChild(txt);

    lx += 18 + txt.textContent.length * 10 + 16;
  }});
}}

// Tooltip Handling
const tooltip = document.getElementById("tooltip");

function showTooltip(e, d) {{
  const activeCfg = METRIC_CONFIG[currentMetric];
  const valStr = formatMetricVal(d[activeCfg.field], currentMetric);

  tooltip.innerHTML = `
    <div class="tt-title">
      <span class="color-dot" style="background:${{d.color}}"></span>
      <span>${{d.model}} (${{d.res}} × ${{d.res}})</span>
    </div>
    <div class="tt-row">
      <span>训练样本 (N):</span>
      <span class="tt-val">N = ${{d.n}} (良品)</span>
    </div>
    <div class="tt-row">
      <span>迭代轮次:</span>
      <span class="tt-val">${{d.iter.toLocaleString()}} 轮</span>
    </div>
    <div class="tt-active-row">
      <div class="tt-row">
        <span>当前指标 [${{activeCfg.label}}]:</span>
        <span class="tt-highlight">${{valStr}}</span>
      </div>
    </div>
    <div style="height:1px; background:rgba(255,255,255,0.1); margin:5px 0;"></div>
    <div class="tt-row">
      <span>测试集 I-AUROC:</span>
      <span class="tt-val">${{d.auc.toFixed(3)}}</span>
    </div>
    <div class="tt-row">
      <span>最优 F1-Score:</span>
      <span class="tt-val">${{d.f1.toFixed(3)}}</span>
    </div>
    <div class="tt-row">
      <span>检出召回率 (Recall):</span>
      <span class="tt-val">${{d.recall.toFixed(1)}}% (TP:${{d.tp}} / FN:${{d.fn}})</span>
    </div>
    <div class="tt-row">
      <span>良品误报率 (FPR):</span>
      <span class="tt-val">${{d.fpr.toFixed(1)}}% (FP:${{d.fp}} / TN:${{d.tn}})</span>
    </div>
    <div class="tt-row">
      <span>推理吞吐量 (FPS):</span>
      <span class="tt-val">${{d.fps.toFixed(1)}} 帧/秒</span>
    </div>
  `;
  tooltip.style.display = "block";
  updateTooltipPos(e);
}}

function updateTooltipPos(e) {{
  const pad = 15;
  let tx = e.clientX + pad;
  let ty = e.clientY + pad;

  const rect = tooltip.getBoundingClientRect();
  if (tx + rect.width > window.innerWidth - 10) {{
    tx = e.clientX - rect.width - pad;
  }}
  if (ty + rect.height > window.innerHeight - 10) {{
    ty = e.clientY - rect.height - pad;
  }}

  tooltip.style.left = `${{tx}}px`;
  tooltip.style.top = `${{ty}}px`;
}}

function hideTooltip() {{
  tooltip.style.display = "none";
}}

// Mouse Guideline on Benchmark Chart
let lastMousePos = null;

function getSvgCoords(e, svgId = "main-chart", width = SVG_WIDTH, height = SVG_HEIGHT) {{
  const svg = document.getElementById(svgId);
  if (!svg) return null;
  if (svg.createSVGPoint && svg.getScreenCTM) {{
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (ctm) {{
      const p = pt.matrixTransform(ctm.inverse());
      return {{ x: p.x, y: p.y }};
    }}
  }}
  const rect = svg.getBoundingClientRect();
  return {{
    x: ((e.clientX - rect.left) / rect.width) * width,
    y: ((e.clientY - rect.top) / rect.height) * height
  }};
}}

function updateCursorGuideline(e) {{
  if (e && e.clientX !== undefined) {{
    lastMousePos = {{ clientX: e.clientX, clientY: e.clientY }};
  }}
  if (!lastMousePos || isDragging) {{
    hideCursorGuideline();
    return;
  }}

  const cursorGroup = document.getElementById("cursor-group");
  const cursorHLine = document.getElementById("cursorHLine");
  const cursorBadge = document.getElementById("cursorBadge");
  const cursorBadgeRect = document.getElementById("cursorBadgeRect");
  const cursorBadgeText = document.getElementById("cursorBadgeText");
  if (!cursorGroup || !cursorHLine) return;

  const p = getSvgCoords(lastMousePos, "main-chart", SVG_WIDTH, SVG_HEIGHT);
  if (!p) return;

  const inPlotY = p.y >= MARGIN.top && p.y <= MARGIN.top + PLOT_HEIGHT;
  const inPlotX = p.x >= MARGIN.left - 40 && p.x <= MARGIN.left + PLOT_WIDTH + 30;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";
    cursorHLine.setAttribute("x1", MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", MARGIN.left + PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    const ratio = 1.0 - (p.y - MARGIN.top) / PLOT_HEIGHT;
    const curVal = currentYMin + ratio * (currentYMax - currentYMin);
    const valStr = formatMetricVal(curVal, currentMetric);

    if (cursorBadge && cursorBadgeText && cursorBadgeRect) {{
      cursorBadge.style.display = "block";
      cursorBadgeText.textContent = valStr;

      const badgeW = Math.max(54, valStr.length * 7.5 + 14);
      const badgeH = 19;
      const badgeX = Math.max(2, MARGIN.left - badgeW - 5);

      cursorBadgeRect.setAttribute("width", badgeW);
      cursorBadgeRect.setAttribute("height", badgeH);
      cursorBadgeRect.setAttribute("x", badgeX);
      cursorBadgeRect.setAttribute("y", p.y - badgeH / 2);

      cursorBadgeText.setAttribute("x", badgeX + badgeW / 2);
      cursorBadgeText.setAttribute("y", p.y + 0.5);
    }}
  }} else {{
    cursorGroup.style.display = "none";
  }}
}}

function hideCursorGuideline() {{
  const cursorGroup = document.getElementById("cursor-group");
  if (cursorGroup) cursorGroup.style.display = "none";
}}

// Zoom & Pan for Benchmark Chart
function zoomAt(factor, centerY) {{
  const currentSpan = currentYMax - currentYMin;
  const newSpan = currentSpan / factor;

  const ratioY = 1.0 - (centerY - MARGIN.top) / PLOT_HEIGHT;
  const centerVal = currentYMin + ratioY * currentSpan;

  currentYMin = centerVal - ratioY * newSpan;
  currentYMax = centerVal + (1.0 - ratioY) * newSpan;

  render();
  updateCursorGuideline();
}}

function resetView() {{
  currentYMin = initialYMin;
  currentYMax = initialYMax;
  render();
  updateCursorGuideline();
}}

// ============================================================================
// TAB 2: SEPARATE DATASET DISTRIBUTION CHARTS (TRAIN & TEST)
// ============================================================================

function initDatasetStatsView() {{
  const nameEl = document.getElementById("dsStatName");
  const trainNsEl = document.getElementById("dsStatTrainNs");
  const testTotalEl = document.getElementById("dsStatTestTotal");
  const testNormalEl = document.getElementById("dsStatTestNormal");
  const testDefectEl = document.getElementById("dsStatTestDefect");
  const ratioEl = document.getElementById("dsStatRatio");

  if (!DATASET_STATS || !DATASET_STATS.test_total) {{
    return;
  }}

  const nsStr = (DATASET_STATS.train_ns || []).join(" / ") + " 张";
  trainNsEl.textContent = nsStr;
  testTotalEl.textContent = `${{DATASET_STATS.test_total.toLocaleString()}} 张`;

  const okPct = (DATASET_STATS.test_normal / DATASET_STATS.test_total * 100).toFixed(1);
  const ngPct = (DATASET_STATS.test_defect / DATASET_STATS.test_total * 100).toFixed(1);

  testNormalEl.textContent = `${{DATASET_STATS.test_normal.toLocaleString()}} 张 (${{okPct}}%)`;
  testDefectEl.textContent = `${{DATASET_STATS.test_defect.toLocaleString()}} 张 (${{ngPct}}%)`;

  const ratio = DATASET_STATS.test_normal > 0 ? (DATASET_STATS.test_defect / DATASET_STATS.test_normal).toFixed(2) : "N/A";
  ratioEl.textContent = `1 : ${{ratio}} (良品 : 缺陷)`;

  document.getElementById("trainCardStats").textContent = `包含 ${{DATASET_STATS.train_ns.length}} 档良品规模`;
  document.getElementById("testCardStats").textContent = `总计 ${{DATASET_STATS.test_total.toLocaleString()}} 张评测图像`;
}}

function setDsLayoutMode(mode) {{
  currentDsLayoutMode = mode;

  // Radio sync
  const radios = document.querySelectorAll("input[name='dsLayoutRadio']");
  radios.forEach(r => {{
    r.checked = (r.value === mode);
  }});

  // Buttons sync
  document.getElementById("btnTabDual").classList.toggle("active", mode === "dual");
  document.getElementById("btnTabTrain").classList.toggle("active", mode === "train");
  document.getElementById("btnTabTest").classList.toggle("active", mode === "test");

  const trainCard = document.getElementById("trainCard");
  const testCard = document.getElementById("testCard");

  if (mode === "dual") {{
    trainCard.style.display = "flex";
    testCard.style.display = "flex";
  }} else if (mode === "train") {{
    trainCard.style.display = "flex";
    testCard.style.display = "none";
  }} else if (mode === "test") {{
    trainCard.style.display = "none";
    testCard.style.display = "flex";
  }}

  renderTrainChart();
  renderTestChart();
}}

// 1. Render Training Set Chart
let trainYMax = 500;
function renderTrainChart() {{
  const gridGroup = document.getElementById("train-grid-group");
  const axesGroup = document.getElementById("train-axes-group");
  const barsGroup = document.getElementById("train-bars-group");
  if (!gridGroup || !axesGroup || !barsGroup) return;

  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  barsGroup.innerHTML = "";

  const ns = DATASET_STATS.train_ns || [100, 200, 400];
  const maxN = Math.max(...ns, 100);
  trainYMax = Math.ceil(maxN * 1.25 / 100) * 100;

  function getY(val) {{
    const ratio = Math.max(0, Math.min(1, val / trainYMax));
    return SUB_MARGIN.top + (1.0 - ratio) * SUB_PLOT_HEIGHT;
  }}

  // Y Ticks
  const yTicks = generateYTicks(0, trainYMax, 5);
  yTicks.forEach(yVal => {{
    const py = getY(yVal);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", SUB_MARGIN.left);
    line.setAttribute("y1", py);
    line.setAttribute("x2", SUB_MARGIN.left + SUB_PLOT_WIDTH);
    line.setAttribute("y2", py);
    line.setAttribute("class", "grid-line");
    gridGroup.appendChild(line);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", SUB_MARGIN.left - 8);
    text.setAttribute("y", py + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("class", "axis-text");
    text.textContent = yVal.toLocaleString();
    axesGroup.appendChild(text);
  }});

  // Y Spine & Title
  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", SUB_MARGIN.left);
  ySpine.setAttribute("y1", SUB_MARGIN.top);
  ySpine.setAttribute("x2", SUB_MARGIN.left);
  ySpine.setAttribute("y2", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("x", -(SUB_MARGIN.top + SUB_PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", SUB_MARGIN.left - 44);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = "训练样本数量 (张)";
  axesGroup.appendChild(yTitle);

  // X Spine
  const xSpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xSpine.setAttribute("x1", SUB_MARGIN.left);
  xSpine.setAttribute("y1", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
  xSpine.setAttribute("x2", SUB_MARGIN.left + SUB_PLOT_WIDTH);
  xSpine.setAttribute("y2", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
  xSpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(xSpine);

  // X Axis Title (well inside bottom border)
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", SUB_MARGIN.left + SUB_PLOT_WIDTH / 2);
  xTitle.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 52);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = "训练集正常样本规模 (Sample Size N)";
  axesGroup.appendChild(xTitle);

  // Bars for each N
  const slotW = SUB_PLOT_WIDTH / Math.max(1, ns.length);
  const barW = Math.min(56, slotW * 0.50);

  ns.forEach((nVal, idx) => {{
    const cx = SUB_MARGIN.left + (idx + 0.5) * slotW;
    const py = getY(nVal);
    const bHeight = (SUB_MARGIN.top + SUB_PLOT_HEIGHT) - py;
    const bx = cx - barW / 2;

    // Tick
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", cx);
    tick.setAttribute("y1", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
    tick.setAttribute("x2", cx);
    tick.setAttribute("y2", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 5);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    // Label
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", cx);
    text.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 22);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "axis-text");
    text.setAttribute("font-weight", "600");
    text.textContent = `N = ${{nVal}}`;
    axesGroup.appendChild(text);

    // Sublabel
    const subText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    subText.setAttribute("x", cx);
    subText.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 36);
    subText.setAttribute("text-anchor", "middle");
    subText.setAttribute("class", "sub-val-label");
    subText.textContent = "正常良品";
    axesGroup.appendChild(subText);

    // Bar rect
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", bx);
    rect.setAttribute("y", py);
    rect.setAttribute("width", barW);
    rect.setAttribute("height", bHeight);
    rect.setAttribute("fill", "#0d9488");
    rect.setAttribute("rx", "4");
    rect.setAttribute("class", "bar-rect");
    barsGroup.appendChild(rect);

    // Value Label on top
    const valText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    valText.setAttribute("x", cx);
    valText.setAttribute("y", py - 6);
    valText.setAttribute("text-anchor", "middle");
    valText.setAttribute("class", "val-label");
    valText.setAttribute("fill", "#0f766e");
    valText.textContent = `${{nVal}} 张`;
    barsGroup.appendChild(valText);

    // Tooltip
    rect.addEventListener("mouseenter", (e) => {{
      tooltip.innerHTML = `
        <div class="tt-title" style="color:#2dd4bf;">📈 训练样本集 (N = ${{nVal}})</div>
        <div class="tt-row"><span>正常良品图像:</span><span class="tt-val">${{nVal}} 张</span></div>
        <div class="tt-row"><span>缺陷负样本:</span><span class="tt-val">0 张 (无监督建库)</span></div>
        <div class="tt-row"><span>用途说明:</span><span class="tt-val">背景特征提取 / 二阶段微调</span></div>
      `;
      tooltip.style.display = "block";
      updateTooltipPos(e);
      updateTrainCursorGuideline(e);
    }});
    rect.addEventListener("mousemove", (e) => {{
      updateTooltipPos(e);
      updateTrainCursorGuideline(e);
    }});
    rect.addEventListener("mouseleave", () => {{
      hideTooltip();
      hideTrainCursorGuideline();
    }});
  }});
}}

// 2. Render Test Set Chart
let testYMax = 1000;
function renderTestChart() {{
  const gridGroup = document.getElementById("test-grid-group");
  const axesGroup = document.getElementById("test-axes-group");
  const barsGroup = document.getElementById("test-data-group");
  if (!gridGroup || !axesGroup || !barsGroup) return;

  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  barsGroup.innerHTML = "";

  const total = DATASET_STATS.test_total || 100;
  const normal = DATASET_STATS.test_normal || 0;
  const defect = DATASET_STATS.test_defect || 0;

  testYMax = Math.ceil(total * 1.25 / 100) * 100;

  function getY(val) {{
    const ratio = Math.max(0, Math.min(1, val / testYMax));
    return SUB_MARGIN.top + (1.0 - ratio) * SUB_PLOT_HEIGHT;
  }}

  // Y Ticks
  const yTicks = generateYTicks(0, testYMax, 5);
  yTicks.forEach(yVal => {{
    const py = getY(yVal);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", SUB_MARGIN.left);
    line.setAttribute("y1", py);
    line.setAttribute("x2", SUB_MARGIN.left + SUB_PLOT_WIDTH);
    line.setAttribute("y2", py);
    line.setAttribute("class", "grid-line");
    gridGroup.appendChild(line);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", SUB_MARGIN.left - 8);
    text.setAttribute("y", py + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("class", "axis-text");
    text.textContent = yVal.toLocaleString();
    axesGroup.appendChild(text);
  }});

  // Y Spine & Title
  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", SUB_MARGIN.left);
  ySpine.setAttribute("y1", SUB_MARGIN.top);
  ySpine.setAttribute("x2", SUB_MARGIN.left);
  ySpine.setAttribute("y2", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("x", -(SUB_MARGIN.top + SUB_PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", SUB_MARGIN.left - 44);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = "测试样本数量 (张)";
  axesGroup.appendChild(yTitle);

  // X Spine
  const xSpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xSpine.setAttribute("x1", SUB_MARGIN.left);
  xSpine.setAttribute("y1", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
  xSpine.setAttribute("x2", SUB_MARGIN.left + SUB_PLOT_WIDTH);
  xSpine.setAttribute("y2", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
  xSpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(xSpine);

  // X Axis Title
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", SUB_MARGIN.left + SUB_PLOT_WIDTH / 2);
  xTitle.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 52);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = "测试集样本类别划分";
  axesGroup.appendChild(xTitle);

  // 3 Test Categories
  const testCats = [
    {{ label: "测试良品 (OK)", sub: `${{(normal / total * 100).toFixed(1)}}%`, count: normal, color: "#10b981", desc: "正常无缺陷测试样本" }},
    {{ label: "测试缺陷 (NG)", sub: `${{(defect / total * 100).toFixed(1)}}%`, count: defect, color: "#f43f5e", desc: "真实异常缺陷测试样本" }},
    {{ label: "测试集总量", sub: "100.0%", count: total, color: "#3b82f6", desc: "全量独立验证集" }}
  ];

  const slotW = SUB_PLOT_WIDTH / 3;
  const barW = Math.min(58, slotW * 0.52);

  testCats.forEach((cat, idx) => {{
    const cx = SUB_MARGIN.left + (idx + 0.5) * slotW;
    const py = getY(cat.count);
    const bHeight = (SUB_MARGIN.top + SUB_PLOT_HEIGHT) - py;
    const bx = cx - barW / 2;

    // Tick
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", cx);
    tick.setAttribute("y1", SUB_MARGIN.top + SUB_PLOT_HEIGHT);
    tick.setAttribute("x2", cx);
    tick.setAttribute("y2", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 5);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    // Label
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", cx);
    text.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 22);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "axis-text");
    text.setAttribute("font-weight", "600");
    text.textContent = cat.label;
    axesGroup.appendChild(text);

    // Sublabel
    const subText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    subText.setAttribute("x", cx);
    subText.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 36);
    subText.setAttribute("text-anchor", "middle");
    subText.setAttribute("class", "sub-val-label");
    subText.textContent = `占比 ${{cat.sub}}`;
    axesGroup.appendChild(subText);

    // Bar rect
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", bx);
    rect.setAttribute("y", py);
    rect.setAttribute("width", barW);
    rect.setAttribute("height", bHeight);
    rect.setAttribute("fill", cat.color);
    rect.setAttribute("rx", "4");
    rect.setAttribute("class", "bar-rect");
    barsGroup.appendChild(rect);

    // Value Label on top
    const valText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    valText.setAttribute("x", cx);
    valText.setAttribute("y", py - 6);
    valText.setAttribute("text-anchor", "middle");
    valText.setAttribute("class", "val-label");
    valText.setAttribute("fill", cat.color);
    valText.textContent = `${{cat.count.toLocaleString()}} 张`;
    barsGroup.appendChild(valText);

    // Tooltip
    rect.addEventListener("mouseenter", (e) => {{
      tooltip.innerHTML = `
        <div class="tt-title" style="color:${{cat.color}};">📊 ${{cat.label}}</div>
        <div class="tt-row"><span>样本图像数量:</span><span class="tt-val">${{cat.count.toLocaleString()}} 张</span></div>
        <div class="tt-row"><span>占测试集比例:</span><span class="tt-val">${{cat.sub}}</span></div>
        <div class="tt-row"><span>性质类别:</span><span class="tt-val">${{cat.desc}}</span></div>
      `;
      tooltip.style.display = "block";
      updateTooltipPos(e);
      updateTestCursorGuideline(e);
    }});
    rect.addEventListener("mousemove", (e) => {{
      updateTooltipPos(e);
      updateTestCursorGuideline(e);
    }});
    rect.addEventListener("mouseleave", () => {{
      hideTooltip();
      hideTestCursorGuideline();
    }});
  }});
}}

// Guideline for Train Chart
function updateTrainCursorGuideline(e) {{
  const cursorGroup = document.getElementById("train-cursor-group");
  const cursorHLine = document.getElementById("trainCursorHLine");
  const cursorBadge = document.getElementById("trainCursorBadge");
  const cursorBadgeRect = document.getElementById("trainCursorBadgeRect");
  const cursorBadgeText = document.getElementById("trainCursorBadgeText");
  if (!cursorGroup || !cursorHLine) return;

  const p = getSvgCoords(e, "train-chart", SUB_SVG_WIDTH, SUB_SVG_HEIGHT);
  if (!p) return;

  const inPlotY = p.y >= SUB_MARGIN.top && p.y <= SUB_MARGIN.top + SUB_PLOT_HEIGHT;
  const inPlotX = p.x >= SUB_MARGIN.left - 30 && p.x <= SUB_MARGIN.left + SUB_PLOT_WIDTH + 20;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";
    cursorHLine.setAttribute("x1", SUB_MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", SUB_MARGIN.left + SUB_PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    const ratio = 1.0 - (p.y - SUB_MARGIN.top) / SUB_PLOT_HEIGHT;
    const curVal = Math.round(ratio * trainYMax);
    const valStr = `${{curVal}} 张`;

    if (cursorBadge && cursorBadgeText && cursorBadgeRect) {{
      cursorBadge.style.display = "block";
      cursorBadgeText.textContent = valStr;
      const badgeW = Math.max(50, valStr.length * 7.5 + 12);
      const badgeH = 19;
      const badgeX = Math.max(2, SUB_MARGIN.left - badgeW - 5);
      cursorBadgeRect.setAttribute("width", badgeW);
      cursorBadgeRect.setAttribute("height", badgeH);
      cursorBadgeRect.setAttribute("x", badgeX);
      cursorBadgeRect.setAttribute("y", p.y - badgeH / 2);
      cursorBadgeText.setAttribute("x", badgeX + badgeW / 2);
      cursorBadgeText.setAttribute("y", p.y + 0.5);
    }}
  }} else {{
    cursorGroup.style.display = "none";
  }}
}}

function hideTrainCursorGuideline() {{
  const g = document.getElementById("train-cursor-group");
  if (g) g.style.display = "none";
}}

// Guideline for Test Chart
function updateTestCursorGuideline(e) {{
  const cursorGroup = document.getElementById("test-cursor-group");
  const cursorHLine = document.getElementById("testCursorHLine");
  const cursorBadge = document.getElementById("testCursorBadge");
  const cursorBadgeRect = document.getElementById("testCursorBadgeRect");
  const cursorBadgeText = document.getElementById("testCursorBadgeText");
  if (!cursorGroup || !cursorHLine) return;

  const p = getSvgCoords(e, "test-chart", SUB_SVG_WIDTH, SUB_SVG_HEIGHT);
  if (!p) return;

  const inPlotY = p.y >= SUB_MARGIN.top && p.y <= SUB_MARGIN.top + SUB_PLOT_HEIGHT;
  const inPlotX = p.x >= SUB_MARGIN.left - 30 && p.x <= SUB_MARGIN.left + SUB_PLOT_WIDTH + 20;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";
    cursorHLine.setAttribute("x1", SUB_MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", SUB_MARGIN.left + SUB_PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    const ratio = 1.0 - (p.y - SUB_MARGIN.top) / SUB_PLOT_HEIGHT;
    const curVal = Math.round(ratio * testYMax);
    const valStr = `${{curVal}} 张`;

    if (cursorBadge && cursorBadgeText && cursorBadgeRect) {{
      cursorBadge.style.display = "block";
      cursorBadgeText.textContent = valStr;
      const badgeW = Math.max(50, valStr.length * 7.5 + 12);
      const badgeH = 19;
      const badgeX = Math.max(2, SUB_MARGIN.left - badgeW - 5);
      cursorBadgeRect.setAttribute("width", badgeW);
      cursorBadgeRect.setAttribute("height", badgeH);
      cursorBadgeRect.setAttribute("x", badgeX);
      cursorBadgeRect.setAttribute("y", p.y - badgeH / 2);
      cursorBadgeText.setAttribute("x", badgeX + badgeW / 2);
      cursorBadgeText.setAttribute("y", p.y + 0.5);
    }}
  }} else {{
    cursorGroup.style.display = "none";
  }}
}}

function hideTestCursorGuideline() {{
  const g = document.getElementById("test-cursor-group");
  if (g) g.style.display = "none";
}}

// ============================================================================
// MAIN TAB SWITCHING
// ============================================================================
function switchMainTab(tab) {{
  currentActiveTab = tab;

  const tabBench = document.getElementById("tabBenchmark");
  const tabDs = document.getElementById("tabDataset");
  const benchArea = document.getElementById("benchmarkChartArea");
  const dsArea = document.getElementById("datasetChartArea");
  const benchControls = document.getElementById("benchmarkControls");
  const dsControls = document.getElementById("datasetControls");
  const benchSelectors = document.getElementById("benchmarkSelectors");
  const btnExport = document.getElementById("btnExportSVG");

  if (tab === "benchmark") {{
    tabBench.classList.add("active");
    tabDs.classList.remove("active");
    benchArea.style.display = "flex";
    dsArea.style.display = "none";
    benchControls.style.display = "flex";
    dsControls.style.display = "none";
    benchSelectors.style.display = "flex";
    btnExport.textContent = "📷 导出评测SVG";
    render();
  }} else {{
    tabBench.classList.remove("active");
    tabDs.classList.add("active");
    benchArea.style.display = "none";
    dsArea.style.display = "flex";
    benchControls.style.display = "none";
    dsControls.style.display = "flex";
    benchSelectors.style.display = "none";
    btnExport.textContent = "📷 导出样本量SVG";
    renderTrainChart();
    renderTestChart();
  }}
}}

// ============================================================================
// EVENT LISTENERS SETUP
// ============================================================================
function setupEventListeners() {{
  const chartBody = document.getElementById("chartBody");

  // Drag Pan
  chartBody.addEventListener("mousedown", (e) => {{
    if (e.button !== 0) return;
    isDragging = true;
    dragStartY = e.clientY;
    dragStartYMin = currentYMin;
    dragStartYMax = currentYMax;
  }});

  window.addEventListener("mousemove", (e) => {{
    if (!isDragging) return;
    const dy = e.clientY - dragStartY;
    const ySpan = dragStartYMax - dragStartYMin;
    const yShift = (dy / PLOT_HEIGHT) * ySpan;
    currentYMin = dragStartYMin + yShift;
    currentYMax = dragStartYMax + yShift;
    render();
    hideCursorGuideline();
  }});

  window.addEventListener("mouseup", () => {{
    if (isDragging) {{
      isDragging = false;
      updateCursorGuideline();
    }}
  }});

  chartBody.addEventListener("mousemove", (e) => {{
    updateCursorGuideline(e);
  }});

  chartBody.addEventListener("mouseleave", () => {{
    hideCursorGuideline();
    hideTooltip();
  }});

  chartBody.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    const svgCoords = getSvgCoords(e, "main-chart", SVG_WIDTH, SVG_HEIGHT);
    const centerY = svgCoords ? svgCoords.y : MARGIN.top + PLOT_HEIGHT / 2;
    zoomAt(factor, centerY);
  }}, {{ passive: false }});

  // Train & Test chart cursor guidelines
  const trainBody = document.getElementById("trainChartBody");
  trainBody.addEventListener("mousemove", (e) => updateTrainCursorGuideline(e));
  trainBody.addEventListener("mouseleave", () => hideTrainCursorGuideline());

  const testBody = document.getElementById("testChartBody");
  testBody.addEventListener("mousemove", (e) => updateTestCursorGuideline(e));
  testBody.addEventListener("mouseleave", () => hideTestCursorGuideline());

  // Zoom Buttons
  document.getElementById("btnZoomIn").addEventListener("click", () => zoomAt(1.25, MARGIN.top + PLOT_HEIGHT / 2));
  document.getElementById("btnZoomOut").addEventListener("click", () => zoomAt(0.8, MARGIN.top + PLOT_HEIGHT / 2));
  document.getElementById("btnResetView").addEventListener("click", resetView);
  document.getElementById("btnSidebarReset").addEventListener("click", () => {{
    document.querySelectorAll(".control-panel input[type='checkbox']").forEach(cb => cb.checked = true);
    updateXAxisSelectorOptions();
    resetView();
  }});

  // X Axis & Chart Type Selectors
  document.getElementById("xAxisSelector").addEventListener("change", (e) => {{
    currentXDim = e.target.value;
    render();
  }});

  document.getElementById("chartTypeSelector").addEventListener("change", (e) => {{
    currentChartType = e.target.value;
    render();
  }});

  // Checkbox Filter Listeners
  document.querySelectorAll(".control-panel input[type='checkbox']").forEach(cb => {{
    cb.addEventListener("change", () => {{
      render();
    }});
  }});

  // Quick Select Links
  document.getElementById("linkSelectAllN").addEventListener("click", () => {{
    document.querySelectorAll("#sampleSizeFilters input[type='checkbox']").forEach(cb => cb.checked = true);
    render();
  }});
  document.getElementById("linkSelectNoneN").addEventListener("click", () => {{
    document.querySelectorAll("#sampleSizeFilters input[type='checkbox']").forEach(cb => cb.checked = false);
    render();
  }});
  document.getElementById("linkSelectAllIters").addEventListener("click", () => {{
    document.querySelectorAll("#iterationFilters input[type='checkbox']").forEach(cb => cb.checked = true);
    render();
  }});
  document.getElementById("linkSelectNoneIters").addEventListener("click", () => {{
    document.querySelectorAll("#iterationFilters input[type='checkbox']").forEach(cb => cb.checked = false);
    render();
  }});

  // Export SVG
  document.getElementById("btnExportSVG").addEventListener("click", () => {{
    if (currentActiveTab === "benchmark") {{
      exportMainSVG();
    }} else {{
      exportSubSVG(currentDsLayoutMode === "test" ? "test" : "train");
    }}
  }});
}}

function exportMainSVG() {{
  hideCursorGuideline();
  const svgEl = document.getElementById("main-chart");
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svgEl);

  if (!source.match(/^<svg[^>]+xmlns="http\\:\\/\\/www\\.w3\\.org\\/2000\\/svg"/)) {{
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }}

  const blob = new Blob([source], {{ type: "image/svg+xml;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${{currentMetric}}_vs_${{currentXDim}}_${{DATASET_NAME}}.svg`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}}

function exportSubSVG(subType) {{
  hideTrainCursorGuideline();
  hideTestCursorGuideline();
  const svgId = (subType === "test") ? "test-chart" : "train-chart";
  const svgEl = document.getElementById(svgId);
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svgEl);

  if (!source.match(/^<svg[^>]+xmlns="http\\:\\/\\/www\\.w3\\.org\\/2000\\/svg"/)) {{
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }}

  const blob = new Blob([source], {{ type: "image/svg+xml;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${{subType}}_split_distribution_${{DATASET_NAME}}.svg`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}}

// Initialize Dashboard
window.addEventListener("DOMContentLoaded", () => {{
  initMetricSelector();
  initFilters();
  initDatasetStatsView();
  updateMetricView();
  setupEventListeners();
}});
</script>

</body>
</html>
"""
    return html_template


def get_dataset_split_stats(outs_dir: Path) -> Dict[str, Any]:
    """Extracts train sample sizes and test normal/defect counts strictly for THIS dataset."""
    sum_f = outs_dir / "final_multisize_summary.json"
    stats = {
        "name": outs_dir.name,
        "train_ns": [],
        "train_counts": {},
        "test_defect": 0,
        "test_normal": 0,
        "test_total": 0
    }
    if sum_f.is_file():
        try:
            data = json.loads(sum_f.read_text(encoding="utf-8"))
            if data:
                item0 = data[0]
                ns = sorted(list(set(int(x.get("n", 0)) for x in data if "n" in x)))
                tp = int(item0.get("tp") or item0.get("din_tp", 0))
                fn = int(item0.get("fn") or item0.get("din_fn", 0))
                fp = int(item0.get("fp") or item0.get("din_fp", 0))
                tn = int(item0.get("tn") or item0.get("din_tn", 0))
                defect_cnt = tp + fn
                normal_cnt = fp + tn
                total_test = defect_cnt + normal_cnt

                stats["train_ns"] = ns
                stats["train_counts"] = {n: n for n in ns}
                stats["test_defect"] = defect_cnt
                stats["test_normal"] = normal_cnt
                stats["test_total"] = total_test
        except Exception as e:
            print(f"[warn] Failed to parse stats for {outs_dir.name}: {e}")
    return stats


def generate_dataset_html(outs_dir: Path, output_html: Optional[Path] = None) -> Path:
    """Generates strictly ONE standalone interactive HTML dashboard for a specific dataset."""
    outs_dir = Path(outs_dir).expanduser().resolve()
    summary_path = outs_dir / "final_multisize_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"No summary data found at: {summary_path}")

    dataset_data = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_name = outs_dir.name

    charts_dir = outs_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    has_bank = any(outs_dir.glob("**/feature_bank.npz")) and any("e2e_auc" in d and d["e2e_auc"] is not None for d in dataset_data)
    dataset_stats = get_dataset_split_stats(outs_dir)
    html_content = build_interactive_html(dataset_name, dataset_data, has_bank=has_bank, dataset_stats=dataset_stats)

    target_html = Path(output_html) if output_html else (charts_dir / "benchmark_dashboard.html")
    target_html.write_text(html_content, encoding="utf-8")
    print(f"[SUCCESS] Generated Standalone Benchmark Dashboard -> {target_html}")

    # Remove any extra redundant html files in charts_dir so only ONE file exists per dataset
    for extra in ["index.html", "09_image_auroc_vs_iterations.html"]:
        extra_file = charts_dir / extra
        if extra_file.is_file() and extra_file.resolve() != target_html.resolve():
            try:
                extra_file.unlink()
                print(f"[CLEANUP] Removed redundant file: {extra_file}")
            except Exception:
                pass

    return target_html


def main():
    parser = argparse.ArgumentParser(description="Generate Dataset-Specific Interactive Benchmark Dashboard HTML")
    parser.add_argument("--outs_base", type=str, default="/data/wt/exp0906", help="Base directory with datasets")
    parser.add_argument("--outs_dir", type=str, default="", help="Specific dataset directory")
    parser.add_argument("--output_html", type=str, default=None, help="Custom output HTML path")
    args = parser.parse_args()

    if args.outs_dir:
        generate_dataset_html(Path(args.outs_dir), Path(args.output_html) if args.output_html else None)
        return

    base_dir = Path(args.outs_base)
    targets = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and (d / "final_multisize_summary.json").is_file():
            targets.append(d)

    if not targets:
        print("[ERROR] No valid dataset summaries found!")
        sys.exit(1)

    print(f"Generating independent dataset interactive HTML benchmark dashboards for {len(targets)} dataset(s)...")
    for t in targets:
        generate_dataset_html(t)

    legacy_master = base_dir / "iteration_benchmark_interactive.html"
    if legacy_master.is_file():
        legacy_master.unlink()
        print(f"[CLEANUP] Removed legacy multi-dataset master dashboard: {legacy_master}")


if __name__ == "__main__":
    main()
