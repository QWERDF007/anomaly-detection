#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import re
import json
import argparse
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
<title>{dataset_name}</title>
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
    --primary: #3b82f6;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    height: 100vh;
    max-height: 100vh;
    overflow: hidden;
    margin: 0;
    padding: 0;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text-primary);
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-height: 100vh;
    overflow: hidden;
  }}

  header {{
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    padding: 8px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    flex-wrap: nowrap;
    gap: 12px;
    flex-shrink: 0;
  }}

  .header-left h1 {{
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
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
    justify-content: flex-end;
    gap: 8px;
    flex-shrink: 0;
    min-width: 80px;
  }}
  #dsSubclassListContainer::-webkit-scrollbar {{
    width: 4px;
  }}
  #dsSubclassListContainer::-webkit-scrollbar-thumb {{
    background: #cbd5e1;
    border-radius: 2px;
  }}
  #dsSubclassListContainer::-webkit-scrollbar-thumb:hover {{
    background: #94a3b8;
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

  .main-layout {{
    display: flex;
    flex: 1;
    min-height: 0;
    padding: 8px 14px;
    gap: 10px;
    width: 100%;
    box-sizing: border-box;
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
  .control-panel::-webkit-scrollbar-thumb:hover {{
    background: #94a3b8;
  }}

  .control-group h3 {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 4px;
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
    background: #f8fafc;
  }}
  .checkbox-item input[type="checkbox"] {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    cursor: pointer;
    accent-color: #2563eb;
  }}

  .color-dot {{
    width: 12px;
    height: 12px;
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

  /* Header Tabs */
  .header-tabs {{
    display: flex;
    align-items: center;
    gap: 4px;
    background: #f1f5f9;
    padding: 3px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin-left: 10px;
  }}
  .header-tabs .tab-btn {{
    padding: 5px 12px;
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

  /* Dataset Split Dual Cards Grid */
  .ds-charts-grid {{
    flex: 1 1 0;
    min-height: 0;
    display: flex;
    gap: 14px;
    padding: 12px 16px;
    background: #ffffff;
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

  .subclass-panel-item {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 0.76rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
    cursor: default;
    transition: all 0.15s ease;
  }}
  .subclass-panel-item:hover {{
    border-color: #cbd5e1;
    background: #f8fafc;
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

  /* Chart Area */
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
    font-size: 1.02rem;
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

  /* Chart Body - NO height:100% to ensure X-axis is not clipped! */
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

  /* Tooltip */
  #tooltip {{
    position: fixed;
    display: none;
    pointer-events: none;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(8px);
    color: white;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
    z-index: 9999;
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

  /* Detail Card */
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

  /* SVG elements */
  .grid-line {{
    stroke: #f1f5f9;
    stroke-width: 1;
    shape-rendering: crispEdges;
  }}
  .category-line {{
    stroke: #e2e8f0;
    stroke-width: 1;
    stroke-dasharray: 4 3;
  }}
  .axis-line {{
    stroke: #94a3b8;
    stroke-width: 1.5;
  }}
  .axis-tick {{
    stroke: #94a3b8;
    stroke-width: 1.2;
  }}
  .axis-text {{
    fill: #475569;
    font-size: 11px;
    font-weight: 500;
    font-family: inherit;
  }}
  .axis-title {{
    fill: #0f172a;
    font-size: 13.5px;
    font-weight: 700;
    font-family: inherit;
  }}
  .bar-item {{
    cursor: pointer;
    transition: filter 0.15s;
  }}
  .bar-item:hover .bar-base {{
    filter: brightness(1.15) drop-shadow(0 2px 6px rgba(0,0,0,0.25));
    stroke: #0f172a !important;
    stroke-width: 1.8px !important;
  }}
  .bar-item:hover .bar-hatch {{
    filter: brightness(1.2);
  }}
  .bar-rect {{
    cursor: pointer;
  }}
  .hatch-swatch {{
    border-radius: 3px;
    border: 1px solid #94a3b8;
    flex-shrink: 0;
    display: inline-block;
    vertical-align: middle;
  }}
  .bar-label {{
    font-size: 10px;
    font-weight: 700;
    pointer-events: none;
    font-family: inherit;
  }}
  .legend-item {{
    cursor: pointer;
    user-select: none;
  }}
  .legend-item:hover text {{
    fill: #0284c7 !important;
  }}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <h1>
      <span>{dataset_name}</span>
    </h1>
    <div class="header-tabs">
      <button class="tab-btn active" id="tabBenchmark" onclick="switchMainTab('benchmark')">📊 算法模型基准评测</button>
      <button class="tab-btn" id="tabDataset" onclick="switchMainTab('dataset')">📦 训练集与测试集数量分布</button>
    </div>
  </div>

  <div class="header-selectors" id="benchmarkSelectors">
    <div class="selector-wrapper">
      <label for="metricSelector">核心评测指标:</label>
      <select id="metricSelector" class="custom-select custom-select-metric"></select>
    </div>

    <div class="selector-wrapper">
      <label for="xAxisSelector">横坐标 (X 轴):</label>
      <select id="xAxisSelector" class="custom-select"></select>
    </div>
  </div>

  <div class="header-selectors" id="datasetSelectors" style="display: none;">
    <div class="selector-wrapper">
      <label for="dsViewSelect">分布视角:</label>
      <select id="dsViewSelect" class="custom-select" style="font-weight: 600; min-width: 200px;">
        <option value="overall" selected>📊 整体分布 (训练集 vs 测试集)</option>
        <option value="subclasses">🏷️ 全量各子类分布 (良品 vs 缺陷)</option>
        <option value="train_subclasses">📈 训练集各子类分布 (各 N 档配额)</option>
        <option value="test_subclasses">🧪 测试集各子类分布 (良品 vs 缺陷)</option>
      </select>
    </div>
  </div>

  <div class="header-actions">
    <button class="btn btn-primary" id="btnExport" title="导出 SVG 矢量图文件">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      导出
    </button>
  </div>
</header>

<div class="main-layout">
  <!-- Control Panel -->
  <aside class="control-panel">
    <!-- Tab 1: Benchmark Controls -->
    <div id="benchmarkControls" style="display: flex; flex-direction: column; gap: 8px;">
    <!-- Model Selection -->
    <div class="control-group">
      <h3>
        <span>模型选择 (Models)</span>
        <div class="quick-links-group">
          <span class="quick-link" id="btnSelectAllModels">全选</span>
          <span class="quick-link-sep">/</span>
          <span class="quick-link" id="btnUnselectAllModels">全不选</span>
        </div>
      </h3>
      <div class="checkbox-list">
{model_checkboxes}
      </div>
    </div>

    <div class="divider"></div>

    <!-- Input Size Selection -->
    <div class="control-group">
      <h3>
        <span>输入尺寸 (Image Sizes)</span>
        <div class="quick-links-group">
          <span class="quick-link" id="btnSelectAllSizes">全选</span>
          <span class="quick-link-sep">/</span>
          <span class="quick-link" id="btnUnselectAllSizes">全不选</span>
        </div>
      </h3>
      <div class="checkbox-list">
        <label class="checkbox-item">
          <input type="checkbox" id="chkSize224" checked data-size="224">
          <svg width="22" height="14" class="hatch-swatch">
            <defs>
              <pattern id="sb-pat-224" width="7" height="7" patternUnits="userSpaceOnUse">
                <circle cx="3.5" cy="3.5" r="1.5" fill="rgba(255,255,255,0.88)" />
              </pattern>
            </defs>
            <rect width="22" height="14" fill="#475569" rx="2" />
            <rect width="22" height="14" fill="url(#sb-pat-224)" rx="2" />
          </svg>
          <span>224 × 224 (圆点纹理)</span>
        </label>
        <label class="checkbox-item">
          <input type="checkbox" id="chkSize448" checked data-size="448">
          <svg width="22" height="14" class="hatch-swatch">
            <defs>
              <pattern id="sb-pat-448" width="8" height="8" patternUnits="userSpaceOnUse">
                <line x1="0" y1="4" x2="8" y2="4" stroke="rgba(255,255,255,0.85)" stroke-width="2.2" />
              </pattern>
            </defs>
            <rect width="22" height="14" fill="#475569" rx="2" />
            <rect width="22" height="14" fill="url(#sb-pat-448)" rx="2" />
          </svg>
          <span>448 × 448 (横线纹理)</span>
        </label>
        <label class="checkbox-item">
          <input type="checkbox" id="chkSize672" checked data-size="672">
          <svg width="22" height="14" class="hatch-swatch">
            <defs>
              <pattern id="sb-pat-672" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="9" stroke="rgba(255,255,255,0.85)" stroke-width="2.6" />
              </pattern>
            </defs>
            <rect width="22" height="14" fill="#475569" rx="2" />
            <rect width="22" height="14" fill="url(#sb-pat-672)" rx="2" />
          </svg>
          <span>672 × 672 (斜线纹理)</span>
        </label>
      </div>
    </div>

    <div class="divider"></div>

    <!-- Sample Size N Selection -->
    <div class="control-group">
      <h3>
        <span>样本量 (Sample Size N)</span>
        <div class="quick-links-group">
          <span class="quick-link" id="btnSelectAllN">全选</span>
          <span class="quick-link-sep">/</span>
          <span class="quick-link" id="btnUnselectAllN">全不选</span>
        </div>
      </h3>
      <div class="checkbox-list" id="sampleNContainer">
      </div>
    </div>

    <div class="divider"></div>

    <!-- Iterations Selection -->
    <div class="control-group" id="itersControlGroup">
      <h3>
        <span>迭代轮次 (Iterations)</span>
        <div class="quick-links-group">
          <span class="quick-link" id="btnSelectAllIters">全选</span>
          <span class="quick-link-sep">/</span>
          <span class="quick-link" id="btnUnselectAllIters">全不选</span>
        </div>
      </h3>
      <div class="checkbox-list" id="itersContainer">
      </div>
    </div>

    <div class="divider" id="itersDivider"></div>

    <!-- Visualization Settings -->
    <div class="control-group">
      <h3>
        <span>Y 轴数值缩放模式</span>
      </h3>
      <div style="margin-top: 6px;">
        <select id="yAxisRangeMode" class="custom-select" style="width: 100%; font-size: 0.85rem; padding: 6px 10px;">
          <option value="adaptive" selected>自适应聚焦 (最大化柱位差异)</option>
          <option value="full">全量标准范围 (从 0 基准起始)</option>
        </select>
      </div>
    </div>
    </div>

    <!-- Tab 2: Dataset Overview Controls -->
    <div id="datasetControls" style="display: none; flex-direction: column; gap: 8px; flex: 1 1 0; min-height: 0; height: 100%; overflow: hidden;">
      <div class="control-group" style="flex-shrink: 0;">
        <h3>当前数据集概览</h3>
        <div class="stat-badge-card">
          <div class="stat-badge-row">
            <span><strong>数据集整体数量:</strong></span>
            <span id="dsStatDatasetTotal" style="font-weight: 700; color: #0f172a;">-</span>
          </div>
          <div class="divider" style="margin: 4px 0;"></div>
          <div class="stat-badge-row">
            <span><strong>训练集:</strong></span>
            <span id="dsStatTrainNs">-</span>
          </div>
          <div class="divider" style="margin: 4px 0;"></div>
          <div class="stat-badge-row">
            <span><strong>测试集:</strong></span>
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
          <div class="divider" id="dsStatSubclassDivider" style="margin: 4px 0; display: none;"></div>
          <div class="stat-badge-row" id="dsStatSubclassRow" style="display: none;">
            <span><strong>子类种类数:</strong></span>
            <span id="dsStatSubclassCount" style="font-weight: 700; color: #6366f1;">-</span>
          </div>
        </div>
      </div>

      <!-- Subclasses Detail Information Panel (Fully fills the remaining height) -->
      <div class="control-group" id="dsSubclassPanelGroup" style="display: none; flex: 1 1 0; min-height: 0; flex-direction: column; gap: 6px; overflow: hidden; margin-top: 2px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;">
          <h3 style="margin: 0; font-size: 0.85rem;">🏷️ 各子类样本明细 (<span id="subclassPanelCount">0</span>)</h3>
          <span style="font-size: 0.70rem; color: #64748b; background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-weight: 500;">真实目录扫描</span>
        </div>
        <div id="dsSubclassListContainer" style="flex: 1 1 0; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; padding-right: 3px;">
          <!-- Dynamically populated by JS -->
        </div>
      </div>
    </div>
  </aside>

  <!-- Chart Container -->
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
            <rect id="clipRect" x="85" y="65" width="1165" height="530" />
          </clipPath>

          <!-- Academic Line & Dot Hatch Patterns for Image Resolutions -->
          <!-- 672x672: +45° Diagonal Stripes /// -->
          <pattern id="pat-672" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="9" stroke="rgba(255,255,255,0.78)" stroke-width="2.6" />
          </pattern>

          <!-- 448x448: Horizontal Stripes --- -->
          <pattern id="pat-448" width="8" height="8" patternUnits="userSpaceOnUse">
            <line x1="0" y1="4" x2="8" y2="4" stroke="rgba(255,255,255,0.80)" stroke-width="2.4" />
          </pattern>

          <!-- 224x224: Circular Polka Dots ::: -->
          <pattern id="pat-224" width="7" height="7" patternUnits="userSpaceOnUse">
            <circle cx="3.5" cy="3.5" r="1.5" fill="rgba(255,255,255,0.88)" />
          </pattern>
        </defs>

        <!-- Static Background Grid & Axes -->
        <g id="grid-group"></g>
        <g id="axes-group"></g>

        <!-- Data Elements (clipped so they don't spill) -->
        <g id="data-group" clip-path="url(#chart-clip)">
          <g id="bars-group"></g>
        </g>

        <!-- Dynamic Header Legends inside SVG -->
        <g id="legend-group"></g>

        <!-- Mouse Guideline (Horizontal Crosshair Dashed Line & Y-Value Badge) -->
        <g id="cursor-group" pointer-events="none" style="display: none;">
          <line id="cursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
          <g id="cursorBadge">
            <rect id="cursorBadgeRect" x="0" y="0" width="56" height="20" rx="3" fill="#0f172a" stroke="#3b82f6" stroke-width="1" />
            <text id="cursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
          </g>
        </g>
      </svg>
    </div>
    </div>

    <!-- 2. Dataset Split Area (Dual Charts vs Subclasses View) -->
    <div id="datasetChartArea" style="display: none; flex-direction: column; width: 100%; height: 100%; min-height: 0; overflow: hidden;">
      <div class="chart-header">
        <div class="title-area">
          <div class="chart-title" id="dsChartTitle">📦 {dataset_name} · 样本量分布看板 (训练集 vs 测试集)</div>
          <div class="chart-stats" id="dsChartStats">展示当前数据集自身的训练规模与测试集构成</div>
        </div>
        <div class="chart-actions" id="dsChartActions"></div>
      </div>

      <div class="ds-charts-grid" id="dsChartsGrid">
        <!-- Left Card: Training Set Chart -->
        <div class="ds-card" id="trainCard">
          <div class="ds-card-header">
            <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a; display: flex; align-items: center; gap: 6px;">
              <span style="display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: #0d9488;"></span>
              📈 训练集样本量分布 (正常良品样本 N)
            </div>
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
        </div>

        <!-- Right Card: Test Set Chart -->
        <div class="ds-card" id="testCard">
          <div class="ds-card-header">
            <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a; display: flex; align-items: center; gap: 6px;">
              <span style="display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: #3b82f6;"></span>
              📊 测试集样本量与构成 (良品 vs 缺陷 vs 总量)
            </div>
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
        </div>
      </div>

      <!-- View 2: Full Dataset Subclasses Distribution (Direct single block, NO nested card) -->
      <div class="chart-body" id="subclassChartBody" style="display: none; flex: 1 1 0; min-height: 0; position: relative;">
        <svg id="subclass-chart" viewBox="0 0 1300 500" preserveAspectRatio="xMidYMid meet" style="width: 100%; height: 100%;">
          <defs>
            <clipPath id="subclass-chart-clip">
              <rect id="subclassClipRect" x="65" y="45" width="1200" height="390" />
            </clipPath>
          </defs>

          <g id="subclass-grid-group"></g>
          <g id="subclass-axes-group"></g>
          <g id="subclass-data-group" clip-path="url(#subclass-chart-clip)">
            <g id="subclass-bars-group"></g>
          </g>

          <!-- Mouse Guideline for Subclass Chart -->
          <g id="subclass-cursor-group" pointer-events="none" style="display: none;">
            <line id="subclassCursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
            <g id="subclassCursorBadge">
              <rect id="subclassCursorBadgeRect" x="0" y="0" width="56" height="20" rx="3" fill="#0f172a" stroke="#6366f1" stroke-width="1" />
              <text id="subclassCursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
            </g>
          </g>
        </svg>
      </div>

      <!-- View 3: Training Set Subclasses Distribution (Direct single block, NO nested card) -->
      <div class="chart-body" id="trainSubclassChartBody" style="display: none; flex: 1 1 0; min-height: 0; position: relative;">
        <svg id="train-subclass-chart" viewBox="0 0 1300 500" preserveAspectRatio="xMidYMid meet" style="width: 100%; height: 100%;">
          <defs>
            <clipPath id="train-subclass-chart-clip">
              <rect id="trainSubclassClipRect" x="65" y="45" width="1200" height="390" />
            </clipPath>
          </defs>

          <g id="train-subclass-grid-group"></g>
          <g id="train-subclass-axes-group"></g>
          <g id="train-subclass-data-group" clip-path="url(#train-subclass-chart-clip)">
            <g id="train-subclass-bars-group"></g>
          </g>

          <!-- Mouse Guideline for Train Subclass Chart -->
          <g id="train-subclass-cursor-group" pointer-events="none" style="display: none;">
            <line id="trainSubclassCursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
            <g id="trainSubclassCursorBadge">
              <rect id="trainSubclassCursorBadgeRect" x="0" y="0" width="56" height="20" rx="3" fill="#0f172a" stroke="#0d9488" stroke-width="1" />
              <text id="trainSubclassCursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
            </g>
          </g>
        </svg>
      </div>

      <!-- View 4: Test Set Subclasses Distribution (Direct single block, NO nested card) -->
      <div class="chart-body" id="testSubclassChartBody" style="display: none; flex: 1 1 0; min-height: 0; position: relative;">
        <svg id="test-subclass-chart" viewBox="0 0 1300 500" preserveAspectRatio="xMidYMid meet" style="width: 100%; height: 100%;">
          <defs>
            <clipPath id="test-subclass-chart-clip">
              <rect id="testSubclassClipRect" x="65" y="45" width="1200" height="390" />
            </clipPath>
          </defs>

          <g id="test-subclass-grid-group"></g>
          <g id="test-subclass-axes-group"></g>
          <g id="test-subclass-data-group" clip-path="url(#test-subclass-chart-clip)">
            <g id="test-subclass-bars-group"></g>
          </g>

          <!-- Mouse Guideline for Test Subclass Chart -->
          <g id="test-subclass-cursor-group" pointer-events="none" style="display: none;">
            <line id="testSubclassCursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
            <g id="testSubclassCursorBadge">
              <rect id="testSubclassCursorBadgeRect" x="0" y="0" width="56" height="20" rx="3" fill="#0f172a" stroke="#3b82f6" stroke-width="1" />
              <text id="testSubclassCursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
            </g>
          </g>
        </svg>
      </div>
    </div>
  </main>
</div>

<!-- Global Tooltip Element -->
<div id="tooltip"></div>

<script>
// Raw Benchmark Data for {dataset_name}
const DATASET_NAME = "{dataset_name}";
const DATASET_DATA = {json_data};
const HAS_BANK = {has_bank_js};

// Color scheme
const COLOR_E2E = "#059669";
const COLOR_DINO = "#2563eb";
const COLOR_PATCH = "#ea580c";

const DATASET_STATS = {dataset_stats_json};
let currentActiveTab = "benchmark";

// Chart Layout Boundaries
const SVG_WIDTH = 1300;
const SVG_HEIGHT = 680;
const MARGIN = {{ top: 65, right: 50, bottom: 85, left: 85 }};

// Sub-charts Layout Boundaries
const SUB_SVG_WIDTH = 650;
const SUB_SVG_HEIGHT = 500;
const SUB_MARGIN = {{ top: 55, right: 35, bottom: 75, left: 65 }};
const SUB_PLOT_WIDTH = SUB_SVG_WIDTH - SUB_MARGIN.left - SUB_MARGIN.right;
const SUB_PLOT_HEIGHT = SUB_SVG_HEIGHT - SUB_MARGIN.top - SUB_MARGIN.bottom;
const PLOT_WIDTH = SVG_WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

// Subclass Chart Layout Boundaries
const SUBCLASS_SVG_WIDTH = 1300;
const SUBCLASS_SVG_HEIGHT = 500;
const SUBCLASS_MARGIN = {{ top: 45, right: 35, bottom: 65, left: 65 }};
const SUBCLASS_PLOT_WIDTH = SUBCLASS_SVG_WIDTH - SUBCLASS_MARGIN.left - SUBCLASS_MARGIN.right;
const SUBCLASS_PLOT_HEIGHT = SUBCLASS_SVG_HEIGHT - SUBCLASS_MARGIN.top - SUBCLASS_MARGIN.bottom;
let subclassYMax = 500;
let currentDsViewMode = "overall"; // "overall" | "subclasses"

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
  fpr: {{
    label: "False Positive Rate (测试集良品误报率 FPR %)",
    title: "测试集未见良品误报率 (FPR = FP / 良品总数, 越低越好)",
    unit: "%",
    digits: 1,
    yLabel: "测试集良品误报率 (%)",
    field: "fpr",
    higherIsBetter: false,
    defaultYRange: [0, 50],
    isRate: true
  }},

  // --- 2. 训练集自检与建库指标 (Training & In-Domain Validation) ---
  clean_fpr: {{
    label: "Clean In-Domain FPR (训练集良品自误报率 %)",
    title: "训练集干净良品自检误报率 (Clean In-Domain FPR, 检验模型自一致性与过拟合, 越低越好)",
    unit: "%",
    digits: 1,
    yLabel: "训练集良品自误报率 (%)",
    field: "clean_fpr",
    higherIsBetter: false,
    defaultYRange: [0, 10],
    isRate: true
  }},
  train_time_m: {{
    label: "Training Time (训练/建库总耗时 min)",
    title: "模型训练迭代 / 特征提取建库总耗时 (Training / Bank Build Time, 越低越好)",
    unit: " min",
    digits: 1,
    yLabel: "训练建库耗时 (min)",
    field: "train_time_m",
    higherIsBetter: false,
    defaultYRange: null,
    isRate: false
  }},

  // --- 3. 硬件与推理效能 (Hardware & Throughput) ---
  lat_ms: {{
    label: "Inference Latency (单图端到端推理时延 ms)",
    title: "GPU 单张图片端到端推理时延 (Latency, 越低越好)",
    unit: " ms",
    digits: 1,
    yLabel: "单图推理时延 (ms)",
    field: "lat_ms",
    higherIsBetter: false,
    defaultYRange: null,
    isRate: false
  }},
  fps: {{
    label: "Inference FPS (实际推理吞吐量 FPS)",
    title: "实际推理吞吐量 (Throughput FPS, 越高越好)",
    unit: " FPS",
    digits: 0,
    yLabel: "推理吞吐量 (FPS)",
    field: "fps",
    higherIsBetter: true,
    defaultYRange: null,
    isRate: false
  }},
  train_vram_gb: {{
    label: "Training VRAM (训练显存占用峰值 GB)",
    title: "GPU 模型训练/特征建库显存峰值 (Training Peak VRAM, 越低越好)",
    unit: " GB",
    digits: 2,
    yLabel: "训练显存占用 (GB)",
    field: "train_vram_gb",
    higherIsBetter: false,
    defaultYRange: [0, 16],
    isRate: false
  }},
  vram_gb: {{
    label: "Inference VRAM (推理显存占用峰值 GB)",
    title: "GPU 单图推理显存峰值 (Inference Peak VRAM, 越低越好)",
    unit: " GB",
    digits: 2,
    yLabel: "推理显存占用 (GB)",
    field: "vram_gb",
    higherIsBetter: false,
    defaultYRange: [0, 8],
    isRate: false
  }}
}};

const NON_ITERATION_METRICS = ["train_vram_gb", "vram_gb", "lat_ms", "fps"];

// Active State
let currentMetric = "auc";
let currentXDim = "iters"; // "iters" or "n"

let currentYMin = 0.80;
let currentYMax = 1.05;
let baseAutoYMin = 0.80;
let baseAutoYMax = 1.05;

let datasetUniqueNs = [];
let datasetUniqueSizes = [];
let datasetUniqueIters = [];

// Pan & Zoom Drag State
let isDragging = false;
let startY = 0;
let dragStartYMin = 0;
let dragStartYMax = 0;

// Initialize
window.addEventListener("DOMContentLoaded", () => {{
  initDataset();
  initDatasetStatsView();
  setupEventListeners();
  if (window.location.hash === "#dataset" || new URLSearchParams(window.location.search).get("tab") === "dataset") {{
    switchMainTab("dataset");
  }}
}});

function updateXAxisSelectorOptions() {{
  const xSel = document.getElementById("xAxisSelector");
  const isNonIter = NON_ITERATION_METRICS.includes(currentMetric);

  if (isNonIter) {{
    currentXDim = "n";
    xSel.innerHTML = '<option value="n" selected>按训练样本量 (Sample Size N)</option>';
  }} else {{
    xSel.innerHTML = `
      <option value="iters" ${{currentXDim === "iters" ? "selected" : ""}}>按迭代轮次 (Iterations)</option>
      <option value="n" ${{currentXDim === "n" ? "selected" : ""}}>按训练样本量 (Sample Size N)</option>
    `;
  }}

  // Hide Iterations selection panel in sidebar when viewing metrics that do not depend on iterations
  const itersGroup = document.getElementById("itersControlGroup");
  if (itersGroup) itersGroup.style.display = isNonIter ? "none" : "block";
  const itersDivider = document.getElementById("itersDivider");
  if (itersDivider) itersDivider.style.display = isNonIter ? "none" : "block";
}}

function initMetricSelector() {{
  const sel = document.getElementById("metricSelector");
  sel.innerHTML = `
    <optgroup label="① 测试集评测指标 (泛化表现 / 缺陷检出与漏报误报)">
      <option value="auc">I-AUROC (测试集图像级 AUC - 区分能力)</option>
      <option value="f1">Optimal F1-Score (测试集最优 F1 - 综合平衡)</option>
      <option value="ap">Average Precision (测试集平均精度 AP)</option>
      <option value="recall">Defect Recall (测试集缺陷检出召回率 % - 越高越好)</option>
      <option value="fnr">Miss Rate / FNR (测试集缺陷漏检率 % - 越低越好)</option>
      <option value="fpr">False Positive Rate (测试集良品误报率 FPR % - 越低越好)</option>
    </optgroup>
    <optgroup label="② 训练集自检与耗时 (良品自检误报 / 建库开销)">
      <option value="clean_fpr">Clean In-Domain FPR (训练集良品自误报率 % - 越低越好)</option>
      <option value="train_time_m">Training Time (模型训练/建库总耗时 min - 越低越好)</option>
      <option value="train_vram_gb">Training VRAM (训练显存占用峰值 GB - 越低越好)</option>
    </optgroup>
    <optgroup label="③ 硬件与推理效能 (按样本量 / 尺寸 / 模型观察)">
      <option value="lat_ms">Inference Latency (单图端到端推理时延 ms - 越低越好)</option>
      <option value="fps">Inference FPS (实际推理吞吐量 FPS - 越高越好)</option>
      <option value="vram_gb">Inference VRAM (单图推理显存占用峰值 GB - 越低越好)</option>
    </optgroup>
  `;
  sel.value = currentMetric;
}}

function initDataset() {{
  datasetUniqueNs = Array.from(new Set(DATASET_DATA.map(r => parseInt(r.n)))).sort((a, b) => a - b);
  datasetUniqueSizes = Array.from(new Set(DATASET_DATA.map(r => parseInt(r.size)))).sort((a, b) => a - b);
  datasetUniqueIters = Array.from(new Set(DATASET_DATA.map(r => parseInt(r.iters || 2000)))).sort((a, b) => a - b);

  // Populate Sample Size checkboxes
  const nContainer = document.getElementById("sampleNContainer");
  nContainer.innerHTML = "";
  datasetUniqueNs.forEach(n => {{
    const label = document.createElement("label");
    label.className = "checkbox-item";
    label.innerHTML = `
      <input type="checkbox" checked data-n="${{n}}">
      <span>N = ${{n}}</span>
    `;
    label.querySelector("input").addEventListener("change", () => {{
      updateYBounds();
      render();
    }});
    nContainer.appendChild(label);
  }});

  // Populate Iterations checkboxes
  const itersContainer = document.getElementById("itersContainer");
  itersContainer.innerHTML = "";
  datasetUniqueIters.forEach(it => {{
    const label = document.createElement("label");
    label.className = "checkbox-item";
    label.innerHTML = `
      <input type="checkbox" checked data-iter="${{it}}">
      <span>${{it.toLocaleString()}} 轮</span>
    `;
    label.querySelector("input").addEventListener("change", () => {{
      updateYBounds();
      render();
    }});
    itersContainer.appendChild(label);
  }});

  initMetricSelector();
  updateXAxisSelectorOptions();
  updateYBounds();
  resetView();
}}

function updateYBounds() {{
  const cfg = METRIC_CONFIG[currentMetric];
  if (!cfg) return;
  const vals = [];

  const selectedModels = new Set();
  const chkE2E = document.getElementById("chkE2E");
  if (chkE2E && chkE2E.checked) selectedModels.add("Two-Stage E2E");
  const chkDino = document.getElementById("chkDino");
  if (chkDino && chkDino.checked) selectedModels.add("Dinomaly2");
  const chkPatch = document.getElementById("chkPatch");
  if (chkPatch && chkPatch.checked) selectedModels.add("PatchCore");

  const selectedSizes = new Set();
  document.querySelectorAll("[data-size]").forEach(cb => {{
    if (cb.checked) selectedSizes.add(parseInt(cb.getAttribute("data-size")));
  }});

  const selectedNs = new Set();
  document.querySelectorAll("[data-n]").forEach(cb => {{
    if (cb.checked) selectedNs.add(parseInt(cb.getAttribute("data-n")));
  }});

  const selectedIters = new Set();
  document.querySelectorAll("[data-iter]").forEach(cb => {{
    if (cb.checked) selectedIters.add(parseInt(cb.getAttribute("data-iter")));
  }});

  DATASET_DATA.forEach(r => {{
    const n = parseInt(r.n);
    const s = parseInt(r.size);
    const it = parseInt(r.iters || 2000);

    if (!selectedNs.has(n) || !selectedSizes.has(s)) return;
    if (!NON_ITERATION_METRICS.includes(currentMetric)) {{
      if (!selectedIters.has(it)) return;
    }}

    if (selectedModels.has("Dinomaly2")) {{
      const v = getPointMetricValue(r, "Dinomaly2", currentMetric);
      if (v !== null && !isNaN(v)) vals.push(v);
    }}
    if (selectedModels.has("Two-Stage E2E")) {{
      const v = getPointMetricValue(r, "Two-Stage E2E", currentMetric);
      if (v !== null && !isNaN(v)) vals.push(v);
    }}
    if (selectedModels.has("PatchCore")) {{
      let allowPatch = true;
      if (!NON_ITERATION_METRICS.includes(currentMetric)) {{
        if (currentXDim === "iters") {{
          allowPatch = selectedIters.has(1000);
        }} else {{
          allowPatch = selectedIters.size > 0;
        }}
      }}
      if (allowPatch) {{
        const v = getPointMetricValue(r, "PatchCore", currentMetric);
        if (v !== null && !isNaN(v)) vals.push(v);
      }}
    }}
  }});

  const mode = document.getElementById("yAxisRangeMode").value;

  if (vals.length > 0) {{
    const minVal = Math.min(...vals);
    const maxVal = Math.max(...vals);

    if (currentMetric === "recall") {{
      if (mode === "adaptive") {{
        const floorVal = Math.max(0.0, Math.floor((minVal - 4) / 5) * 5);
        baseAutoYMin = floorVal;
        baseAutoYMax = Math.min(100.0, Math.ceil(maxVal + 1));
        if (baseAutoYMax <= baseAutoYMin) baseAutoYMax = 100.0;
      }} else {{
        baseAutoYMin = 0.0;
        baseAutoYMax = 105.0;
      }}
    }} else if (currentMetric === "fnr" || currentMetric === "fpr" || currentMetric === "clean_fpr") {{
      baseAutoYMin = 0.0;
      baseAutoYMax = Math.min(100.0, Math.max(5.0, Math.ceil(maxVal * 1.25)));
    }} else if (["auc", "f1", "ap"].includes(currentMetric)) {{
      if (mode === "adaptive") {{
        // Distinguish high values near 1.0 (e.g. 0.85 ~ 1.0)
        const floorVal = Math.max(0.0, Math.floor((minVal - 0.04) * 10) / 10);
        baseAutoYMin = floorVal;
        baseAutoYMax = 1.02;
      }} else {{
        baseAutoYMin = 0.0;
        baseAutoYMax = 1.05;
      }}
    }} else {{
      baseAutoYMin = 0.0;
      baseAutoYMax = maxVal > 0 ? maxVal * 1.22 : 10.0;
    }}
  }} else {{
    baseAutoYMin = cfg.defaultYRange ? cfg.defaultYRange[0] : 0.0;
    baseAutoYMax = cfg.defaultYRange ? cfg.defaultYRange[1] : 1.0;
  }}

  currentYMin = baseAutoYMin;
  currentYMax = baseAutoYMax;
}}

function getPointMetricValue(r, model, metric) {{
  if (metric === "auc") {{
    if (model === "Dinomaly2") return parseFloat(r.din_auc || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_auc || 0);
    if (model === "PatchCore") return parseFloat(r.pat_auc || 0);
  }}
  if (metric === "f1") {{
    if (model === "Dinomaly2") return parseFloat(r.din_f1 || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_f1 || 0);
    if (model === "PatchCore") return parseFloat(r.pat_f1 || 0);
  }}
  if (metric === "ap") {{
    if (model === "Dinomaly2") return parseFloat(r.din_ap || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_ap || 0);
    if (model === "PatchCore") return parseFloat(r.pat_ap || 0);
  }}
  if (metric === "tp") {{
    if (model === "Dinomaly2") return parseInt(r.din_tp || 0);
    if (model === "Two-Stage E2E") return parseInt(r.e2e_tp || 0);
    if (model === "PatchCore") return parseInt(r.pat_tp || 0);
  }}
  if (metric === "fn") {{
    if (model === "Dinomaly2") return parseInt(r.din_fn || 0);
    if (model === "Two-Stage E2E") return parseInt(r.e2e_fn || 0);
    if (model === "PatchCore") return parseInt(r.pat_fn || 0);
  }}
  if (metric === "fnr") {{
    let tp = 0, fn = 0;
    if (model === "Dinomaly2") {{ tp = parseInt(r.din_tp || 0); fn = parseInt(r.din_fn || 0); }}
    if (model === "Two-Stage E2E") {{ tp = parseInt(r.e2e_tp || 0); fn = parseInt(r.e2e_fn || 0); }}
    if (model === "PatchCore") {{ tp = parseInt(r.pat_tp || 0); fn = parseInt(r.pat_fn || 0); }}
    const total = tp + fn;
    return total > 0 ? parseFloat((fn / total * 100).toFixed(2)) : 0;
  }}
  if (metric === "recall") {{
    let tp = 0, fn = 0;
    if (model === "Dinomaly2") {{ tp = parseInt(r.din_tp || 0); fn = parseInt(r.din_fn || 0); }}
    if (model === "Two-Stage E2E") {{ tp = parseInt(r.e2e_tp || 0); fn = parseInt(r.e2e_fn || 0); }}
    if (model === "PatchCore") {{ tp = parseInt(r.pat_tp || 0); fn = parseInt(r.pat_fn || 0); }}
    const total = tp + fn;
    return total > 0 ? parseFloat((tp / total * 100).toFixed(2)) : 0;
  }}
  if (metric === "fp") {{
    if (model === "Dinomaly2") return parseInt(r.din_fp || 0);
    if (model === "Two-Stage E2E") return parseInt(r.e2e_fp || 0);
    if (model === "PatchCore") return parseInt(r.pat_fp || 0);
  }}
  if (metric === "fpr") {{
    let fp = 0, tn = 0;
    if (model === "Dinomaly2") {{ fp = parseInt(r.din_fp || 0); tn = parseInt(r.din_tn || 0); }}
    if (model === "Two-Stage E2E") {{ fp = parseInt(r.e2e_fp || 0); tn = parseInt(r.e2e_tn || 0); }}
    if (model === "PatchCore") {{ fp = parseInt(r.pat_fp || 0); tn = parseInt(r.pat_tn || 0); }}
    const total = fp + tn;
    return total > 0 ? parseFloat((fp / total * 100).toFixed(2)) : 0;
  }}
  if (metric === "clean_fpr") {{
    if (model === "Dinomaly2") return parseFloat(r.din_clean_fpr || 0) * 100;
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_clean_fpr || 0) * 100;
    if (model === "PatchCore") return parseFloat(r.pat_clean_fpr || 0) * 100;
  }}
  if (metric === "lat_ms") {{
    if (model === "Dinomaly2") return parseFloat(r.din_lat_ms || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_lat_ms || 0);
    if (model === "PatchCore") return parseFloat(r.pat_lat_ms || 0);
  }}
  if (metric === "fps") {{
    if (model === "Dinomaly2") return parseFloat(r.din_fps || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_fps || r.fps || 0);
    if (model === "PatchCore") return parseFloat(r.pat_fps || 0);
  }}
  if (metric === "train_time_m") {{
    if (model === "Dinomaly2") return parseFloat(r.din_train_time_m || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_sec ? (r.e2e_sec / 60) : (r.din_train_time_m || 0));
    if (model === "PatchCore") return parseFloat(r.pat_train_time_m || (r.pat_train_time_s ? r.pat_train_time_s / 60 : 0));
  }}
  if (metric === "train_vram_gb") {{
    if (model === "Dinomaly2") return parseFloat(r.din_train_vram_gb || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_train_vram_gb || r.din_train_vram_gb || 0);
    if (model === "PatchCore") return parseFloat(r.pat_train_vram_gb || 0);
  }}
  if (metric === "vram_gb") {{
    if (model === "Dinomaly2") return parseFloat(r.din_vram_gb || 0);
    if (model === "Two-Stage E2E") return parseFloat(r.e2e_vram_gb || 0);
    if (model === "PatchCore") return parseFloat(r.pat_vram_gb || 0);
  }}
  return 0;
}}

function extractItemMetrics(r, model) {{
  if (model === "Two-Stage E2E") {{
    const tp = parseInt(r.e2e_tp || 0);
    const fn = parseInt(r.e2e_fn || 0);
    const fp = parseInt(r.e2e_fp || 0);
    const tn = parseInt(r.e2e_tn || 0);
    const totalDefects = tp + fn;
    const totalNormals = fp + tn;
    const recall = totalDefects > 0 ? (tp / totalDefects * 100) : 0;
    const fnr = totalDefects > 0 ? (fn / totalDefects * 100) : 0;
    const fpr = totalNormals > 0 ? (fp / totalNormals * 100) : 0;
    return {{
      auc: parseFloat(r.e2e_auc || 0),
      f1: parseFloat(r.e2e_f1 || 0),
      ap: parseFloat(r.e2e_ap || 0),
      tp: tp,
      fn: fn,
      fp: fp,
      tn: tn,
      recall: recall,
      fnr: fnr,
      fpr: fpr,
      clean_fpr: parseFloat(r.e2e_clean_fpr || 0) * 100,
      lat_ms: parseFloat(r.e2e_lat_ms || 0),
      fps: parseFloat(r.e2e_fps || r.fps || 0),
      train_time_m: parseFloat(r.e2e_sec ? (r.e2e_sec / 60) : (r.din_train_time_m || 0)),
      vram_gb: parseFloat(r.e2e_vram_gb || 0),
      train_vram_gb: parseFloat(r.e2e_train_vram_gb || r.din_train_vram_gb || 0)
    }};
  }} else if (model === "PatchCore") {{
    const tp = parseInt(r.pat_tp || 0);
    const fn = parseInt(r.pat_fn || 0);
    const fp = parseInt(r.pat_fp || 0);
    const tn = parseInt(r.pat_tn || 0);
    const totalDefects = tp + fn;
    const totalNormals = fp + tn;
    const recall = totalDefects > 0 ? (tp / totalDefects * 100) : 0;
    const fnr = totalDefects > 0 ? (fn / totalDefects * 100) : 0;
    const fpr = totalNormals > 0 ? (fp / totalNormals * 100) : 0;
    return {{
      auc: parseFloat(r.pat_auc || 0),
      f1: parseFloat(r.pat_f1 || 0),
      ap: parseFloat(r.pat_ap || 0),
      tp: tp,
      fn: fn,
      fp: fp,
      tn: tn,
      recall: recall,
      fnr: fnr,
      fpr: fpr,
      clean_fpr: parseFloat(r.pat_clean_fpr || 0) * 100,
      lat_ms: parseFloat(r.pat_lat_ms || 0),
      fps: parseFloat(r.pat_fps || 0),
      train_time_m: parseFloat(r.pat_train_time_m || (r.pat_train_time_s ? r.pat_train_time_s / 60 : 0)),
      vram_gb: parseFloat(r.pat_vram_gb || 0),
      train_vram_gb: parseFloat(r.pat_train_vram_gb || 0)
    }};
  }} else {{
    const tp = parseInt(r.din_tp || 0);
    const fn = parseInt(r.din_fn || 0);
    const fp = parseInt(r.din_fp || 0);
    const tn = parseInt(r.din_tn || 0);
    const totalDefects = tp + fn;
    const totalNormals = fp + tn;
    const recall = totalDefects > 0 ? (tp / totalDefects * 100) : 0;
    const fnr = totalDefects > 0 ? (fn / totalDefects * 100) : 0;
    const fpr = totalNormals > 0 ? (fp / totalNormals * 100) : 0;
    return {{
      auc: parseFloat(r.din_auc || 0),
      f1: parseFloat(r.din_f1 || 0),
      ap: parseFloat(r.din_ap || 0),
      tp: tp,
      fn: fn,
      fp: fp,
      tn: tn,
      recall: recall,
      fnr: fnr,
      fpr: fpr,
      clean_fpr: parseFloat(r.din_clean_fpr || 0) * 100,
      lat_ms: parseFloat(r.din_lat_ms || 0),
      fps: parseFloat(r.din_fps || 0),
      train_time_m: parseFloat(r.din_train_time_m || 0),
      vram_gb: parseFloat(r.din_vram_gb || 0),
      train_vram_gb: parseFloat(r.din_train_vram_gb || 0)
    }};
  }}
}}

function updateTitleAndStats(count) {{
  const activeCfg = METRIC_CONFIG[currentMetric];
  const xDimTitle = currentXDim === "iters" ? "训练迭代轮次 (Iterations)" : "正常训练样本量 (N)";
  document.getElementById("chartTitle").textContent = `${{activeCfg.label}} vs. ${{xDimTitle}} [分组柱状图]`;
  document.getElementById("chartStats").textContent = `当前展示 ${{count}} 个有效实验柱位 | 视窗范围: Y=[${{formatMetricVal(currentYMin, currentMetric)}}, ${{formatMetricVal(currentYMax, currentMetric)}}]`;
}}

function valToY(val) {{
  const span = currentYMax - currentYMin;
  if (span <= 0) return MARGIN.top + PLOT_HEIGHT;
  const ratio = (val - currentYMin) / span;
  return MARGIN.top + (1.0 - ratio) * PLOT_HEIGHT;
}}

function formatMetricVal(val, key) {{
  const cfg = METRIC_CONFIG[key];
  if (val === null || val === undefined || isNaN(val)) return "N/A";
  if (cfg.unit === "%") return val.toFixed(cfg.digits !== undefined ? cfg.digits : 1) + "%";
  if (cfg.digits === 0) return Math.round(val) + (cfg.unit || "");
  return val.toFixed(cfg.digits) + (cfg.unit || "");
}}

function render() {{
  const clipRect = document.getElementById("clipRect");
  clipRect.setAttribute("x", MARGIN.left);
  clipRect.setAttribute("y", MARGIN.top);
  clipRect.setAttribute("width", PLOT_WIDTH);
  clipRect.setAttribute("height", PLOT_HEIGHT);

  const selectedModels = new Set();
  const chkE2E = document.getElementById("chkE2E");
  if (chkE2E && chkE2E.checked) selectedModels.add("Two-Stage E2E");
  const chkDino = document.getElementById("chkDino");
  if (chkDino && chkDino.checked) selectedModels.add("Dinomaly2");
  const chkPatch = document.getElementById("chkPatch");
  if (chkPatch && chkPatch.checked) selectedModels.add("PatchCore");

  const selectedSizes = new Set();
  document.querySelectorAll("[data-size]").forEach(cb => {{
    if (cb.checked) selectedSizes.add(parseInt(cb.getAttribute("data-size")));
  }});

  const selectedNs = new Set();
  document.querySelectorAll("[data-n]").forEach(cb => {{
    if (cb.checked) selectedNs.add(parseInt(cb.getAttribute("data-n")));
  }});

  const selectedIters = new Set();
  document.querySelectorAll("[data-iter]").forEach(cb => {{
    if (cb.checked) selectedIters.add(parseInt(cb.getAttribute("data-iter")));
  }});

  // Clear bars group
  document.getElementById("bars-group").innerHTML = "";

  renderAxesAndGrid();
  renderBarChart(selectedModels, selectedSizes, selectedNs, selectedIters);
}}

function renderBarChart(selectedModels, selectedSizes, selectedNs, selectedIters) {{
  const barsGroup = document.getElementById("bars-group");
  barsGroup.innerHTML = "";

  // Categories along X-axis
  const categories = currentXDim === "n"
    ? datasetUniqueNs.filter(n => selectedNs.has(n))
    : datasetUniqueIters.filter(it => selectedIters.has(it));

  if (categories.length === 0 || selectedModels.size === 0 || selectedSizes.size === 0 || selectedNs.size === 0) {{
    updateTitleAndStats(0);
    return;
  }}

  if (currentXDim === "n" && !NON_ITERATION_METRICS.includes(currentMetric) && selectedIters.size === 0) {{
    updateTitleAndStats(0);
    return;
  }}

  const slotWidth = PLOT_WIDTH / categories.length;
  let totalRenderedBars = 0;

  // Active models, sizes, N, and iters in consistent visual order
  const modelOrder = ["Two-Stage E2E", "Dinomaly2", "PatchCore"].filter(m => selectedModels.has(m));
  const sizeOrder = [224, 448, 672].filter(s => selectedSizes.has(s));
  const nOrder = datasetUniqueNs.filter(n => selectedNs.has(n));
  const iterOrder = datasetUniqueIters.filter(it => selectedIters.has(it));

  categories.forEach((catVal, catIdx) => {{
    const barItems = [];

    if (currentXDim === "iters") {{
      // X-Axis is Iteration (catVal is iters)
      modelOrder.forEach(model => {{
        if (model === "PatchCore" && catVal !== 1000) return;
        nOrder.forEach(n => {{
          sizeOrder.forEach(sz => {{
            const matchedRow = DATASET_DATA.find(d => parseInt(d.iters || 2000) === catVal && parseInt(d.size) === sz && parseInt(d.n) === n);
            if (!matchedRow) return;

            const yVal = getPointMetricValue(matchedRow, model, currentMetric);
            if (yVal === null || isNaN(yVal)) return;

            const color = model === "Two-Stage E2E" ? COLOR_E2E : (model === "Dinomaly2" ? COLOR_DINO : COLOR_PATCH);
            const opacity = 0.95;

            const mMetrics = extractItemMetrics(matchedRow, model);
            barItems.push({{
              model: model,
              size: sz,
              n: parseInt(matchedRow.n),
              iters: parseInt(matchedRow.iters || 2000),
              y_val: yVal,
              color: color,
              opacity: opacity,
              raw: matchedRow,
              ...mMetrics
            }});
          }});
        }});
      }});
    }} else {{
      // X-Axis is Sample Size N (catVal is n)
      if (NON_ITERATION_METRICS.includes(currentMetric)) {{
        // Hardware & throughput metrics don't vary with iterations
        modelOrder.forEach(model => {{
          sizeOrder.forEach(sz => {{
            const matchedRow = DATASET_DATA.find(d => parseInt(d.n) === catVal && parseInt(d.size) === sz);
            if (!matchedRow) return;

            const yVal = getPointMetricValue(matchedRow, model, currentMetric);
            if (yVal === null || isNaN(yVal)) return;

            const color = model === "Two-Stage E2E" ? COLOR_E2E : (model === "Dinomaly2" ? COLOR_DINO : COLOR_PATCH);
            const opacity = 0.95;

            const mMetrics = extractItemMetrics(matchedRow, model);
            barItems.push({{
              model: model,
              size: sz,
              n: parseInt(matchedRow.n),
              iters: parseInt(matchedRow.iters || 2000),
              y_val: yVal,
              color: color,
              opacity: opacity,
              raw: matchedRow,
              ...mMetrics
            }});
          }});
        }});
      }} else {{
        // Accuracy and duration metrics vary with iterations
        modelOrder.forEach(model => {{
          iterOrder.forEach(it => {{
            if (model === "PatchCore" && it !== 1000) return;
            sizeOrder.forEach(sz => {{
              const matchedRow = DATASET_DATA.find(d => parseInt(d.n) === catVal && parseInt(d.size) === sz && parseInt(d.iters || 2000) === it);
              if (!matchedRow) return;

              const yVal = getPointMetricValue(matchedRow, model, currentMetric);
              if (yVal === null || isNaN(yVal)) return;

              const color = model === "Two-Stage E2E" ? COLOR_E2E : (model === "Dinomaly2" ? COLOR_DINO : COLOR_PATCH);
              const opacity = 0.95;

              const mMetrics = extractItemMetrics(matchedRow, model);
              barItems.push({{
                model: model,
                size: sz,
                n: parseInt(matchedRow.n),
                iters: parseInt(matchedRow.iters || 2000),
                y_val: yVal,
                color: color,
                opacity: opacity,
                raw: matchedRow,
                ...mMetrics
              }});
            }});
          }});
        }});
      }}
    }}

    if (barItems.length === 0) return;
    totalRenderedBars += barItems.length;

    // Position bars centered in category slot
    const M = barItems.length;
    const maxGroupWidth = slotWidth * 0.88;
    const desiredGap = M > 8 ? 2 : (M > 4 ? 3 : 4);
    const barW = Math.min(34, Math.max(3, (maxGroupWidth - (M - 1) * desiredGap) / M));
    const barGap = M > 1 ? desiredGap : 0;
    const totalGroupW = M * barW + (M - 1) * barGap;
    const slotCenterX = MARGIN.left + (catIdx + 0.5) * slotWidth;
    const groupStartX = slotCenterX - totalGroupW / 2;

    const ySpan = (currentYMax - currentYMin) || 1.0;

    barItems.forEach((item, itemIdx) => {{
      const bx = groupStartX + itemIdx * (barW + barGap);
      const bHeight = Math.max(2, ((item.y_val - currentYMin) / ySpan) * PLOT_HEIGHT);
      const by = MARGIN.top + PLOT_HEIGHT - bHeight;

      const barG = document.createElementNS("http://www.w3.org/2000/svg", "g");
      barG.setAttribute("class", "bar-item");

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", bx);
      rect.setAttribute("y", by);
      rect.setAttribute("width", barW);
      rect.setAttribute("height", bHeight);
      rect.setAttribute("fill", item.color);
      rect.setAttribute("fill-opacity", item.opacity);
      rect.setAttribute("stroke", "#ffffff");
      rect.setAttribute("stroke-width", "1.2");
      rect.setAttribute("rx", "3");
      rect.setAttribute("ry", "3");
      rect.setAttribute("class", "bar-base bar-rect");

      rect.addEventListener("mouseenter", (e) => showTooltip(e, item));
      rect.addEventListener("mousemove", (e) => updateTooltipPos(e));
      rect.addEventListener("mouseleave", hideTooltip);
      rect.addEventListener("click", () => showDetailCard(item));

      barG.appendChild(rect);

      // Hatch pattern overlay rect
      const patRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      patRect.setAttribute("x", bx);
      patRect.setAttribute("y", by);
      patRect.setAttribute("width", barW);
      patRect.setAttribute("height", bHeight);
      patRect.setAttribute("fill", `url(#pat-${{item.size}})`);
      patRect.setAttribute("rx", "3");
      patRect.setAttribute("ry", "3");
      patRect.setAttribute("class", "bar-hatch bar-rect");
      patRect.setAttribute("pointer-events", "none");
      barG.appendChild(patRect);

      barsGroup.appendChild(barG);

      // Label text above bar
      if (barW >= 10 && bHeight > 3) {{
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", bx + barW / 2);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("class", "bar-label");
        text.setAttribute("fill", item.model === "Two-Stage E2E" ? "#047857" : (item.model === "Dinomaly2" ? "#1d4ed8" : "#c2410c"));

        let valLabel = "";
        const cfg = METRIC_CONFIG[currentMetric];
        if (["auc", "f1", "ap"].includes(currentMetric)) {{
          valLabel = item.y_val.toFixed(3);
        }} else if (cfg.unit === "%" || ["recall", "fnr", "fpr", "clean_fpr"].includes(currentMetric)) {{
          valLabel = item.y_val.toFixed(cfg.digits !== undefined ? cfg.digits : 1) + "%";
        }} else if (currentMetric === "train_time_m") {{
          valLabel = item.y_val < 1 ? item.y_val.toFixed(2) + "m" : item.y_val.toFixed(1) + "m";
        }} else if (["vram_gb", "train_vram_gb"].includes(currentMetric)) {{
          valLabel = item.y_val.toFixed(2) + "G";
        }} else if (currentMetric === "lat_ms") {{
          valLabel = item.y_val.toFixed(1) + "ms";
        }} else if (["fps", "tp", "fp", "fn", "tn"].includes(currentMetric) || cfg.digits === 0) {{
          valLabel = Math.round(item.y_val).toString();
        }} else {{
          valLabel = item.y_val.toFixed(cfg.digits);
        }}

        const showNTag = currentXDim === "iters" && nOrder.length > 1;
        const showIterTag = currentXDim === "n" && iterOrder.length > 1 && !NON_ITERATION_METRICS.includes(currentMetric);

        if (showNTag || showIterTag) {{
          const subText = showNTag ? `N=${{item.n}}` : (item.iters >= 1000 ? `${{item.iters / 1000}}k轮` : `${{item.iters}}轮`);
          if (barW >= 12) {{
            const valFontSize = barW >= 20 ? "10px" : (barW >= 15 ? "9px" : "8px");
            const subFontSize = barW >= 20 ? "8.5px" : (barW >= 15 ? "7.5px" : "7px");
            const dyVal = barW >= 20 ? 11 : 9.5;

            text.setAttribute("y", Math.max(MARGIN.top + 16, by - 15));
            text.setAttribute("font-size", valFontSize);
            text.innerHTML = `
              <tspan x="${{bx + barW / 2}}" dy="0">${{valLabel}}</tspan>
              <tspan x="${{bx + barW / 2}}" dy="${{dyVal}}" font-size="${{subFontSize}}" font-weight="600" fill="#64748b">${{subText}}</tspan>
            `;
          }} else {{
            text.setAttribute("y", Math.max(MARGIN.top + 10, by - 5));
            text.setAttribute("font-size", barW < 14 ? "8px" : "9px");
            text.textContent = valLabel;
          }}
        }} else {{
          text.setAttribute("y", Math.max(MARGIN.top + 10, by - 5));
          text.setAttribute("font-size", barW < 16 ? "9px" : "10px");
          text.textContent = valLabel;
        }}

        barsGroup.appendChild(text);
      }}
    }});
  }});

  updateTitleAndStats(totalRenderedBars);
}}

function renderAxesAndGrid() {{
  const gridGroup = document.getElementById("grid-group");
  const axesGroup = document.getElementById("axes-group");
  const legendGroup = document.getElementById("legend-group");
  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  legendGroup.innerHTML = "";

  // 1. X Axis (Categorical Slots)
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

    // Category Label
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
      divLine.setAttribute("class", "category-line");
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

  // X Axis Title
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", MARGIN.left + PLOT_WIDTH / 2);
  xTitle.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 52);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = currentXDim === "iters" ? "训练迭代轮次 (Iterations)" : "正常训练样本量 (Sample Size N)";
  axesGroup.appendChild(xTitle);

  // 2. Y Axis
  const yTicks = generateYTicks(currentYMin, currentYMax, 6);
  yTicks.forEach(yVal => {{
    const py = valToY(yVal);
    if (py < MARGIN.top - 2 || py > MARGIN.top + PLOT_HEIGHT + 2) return;

    const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gridLine.setAttribute("x1", MARGIN.left);
    gridLine.setAttribute("y1", py);
    gridLine.setAttribute("x2", MARGIN.left + PLOT_WIDTH);
    gridLine.setAttribute("y2", py);
    gridLine.setAttribute("class", "grid-line");
    gridGroup.appendChild(gridLine);

    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", MARGIN.left - 5);
    tick.setAttribute("y1", py);
    tick.setAttribute("x2", MARGIN.left);
    tick.setAttribute("y2", py);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", MARGIN.left - 10);
    text.setAttribute("y", py + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("class", "axis-text");
    text.textContent = formatTickVal(yVal, currentMetric);
    axesGroup.appendChild(text);
  }});

  // Y Axis Spine Line
  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", MARGIN.left);
  ySpine.setAttribute("y1", MARGIN.top);
  ySpine.setAttribute("x2", MARGIN.left);
  ySpine.setAttribute("y2", MARGIN.top + PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  // Y Axis Title
  const activeCfg = METRIC_CONFIG[currentMetric];
  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("x", -(MARGIN.top + PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", MARGIN.left - 52);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = activeCfg.yLabel;
  axesGroup.appendChild(yTitle);

  // 3. Top Academic Legend inside SVG
  renderTopHeaderLegend(legendGroup);
}}

function generateYTicks(min, max, count = 6) {{
  const span = max - min;
  if (span <= 0) return [min];
  const rawStep = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;

  let step;
  if (norm < 1.5) step = 1 * mag;
  else if (norm < 3.0) step = 2 * mag;
  else if (norm < 7.0) step = 5 * mag;
  else step = 10 * mag;

  const ticks = [];
  const start = Math.ceil(min / step) * step;
  for (let t = start; t <= max + step * 0.001; t += step) {{
    ticks.push(t);
  }}
  return ticks;
}}

function formatTickVal(val, key) {{
  const cfg = METRIC_CONFIG[key];
  if (cfg.unit === "%") {{
    return (Number.isInteger(val) ? val.toString() : val.toFixed(1)) + "%";
  }}
  if (cfg.digits === 0) return Math.round(val).toString();
  return val.toFixed(cfg.digits);
}}

function renderTopHeaderLegend(group) {{
  const models = [];
  if (HAS_BANK) {{
    models.push({{ id: "chkE2E", name: "二阶段端到端 (Two-Stage E2E)", color: COLOR_E2E }});
    models.push({{ id: "chkDino", name: "Dinomaly2 (单阶段)", color: COLOR_DINO }});
  }} else {{
    models.push({{ id: "chkDino", name: "Dinomaly2", color: COLOR_DINO }});
  }}
  models.push({{ id: "chkPatch", name: "PatchCore (基线)", color: COLOR_PATCH }});

  let curX = MARGIN.left;
  models.forEach(m => {{
    const cb = document.getElementById(m.id);
    const isChecked = cb ? cb.checked : true;

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "legend-item");
    if (!isChecked) g.setAttribute("opacity", "0.35");
    g.addEventListener("click", () => {{
      if (cb) {{
        cb.checked = !cb.checked;
        updateYBounds();
        render();
      }}
    }});

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", curX);
    rect.setAttribute("y", MARGIN.top - 28);
    rect.setAttribute("width", 12);
    rect.setAttribute("height", 12);
    rect.setAttribute("rx", "2.5");
    rect.setAttribute("fill", m.color);
    g.appendChild(rect);

    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", curX + 17);
    txt.setAttribute("y", MARGIN.top - 18);
    txt.setAttribute("class", "axis-text");
    txt.setAttribute("font-weight", "600");
    txt.setAttribute("fill", "#1e293b");
    txt.textContent = m.name;
    g.appendChild(txt);

    group.appendChild(g);
    curX += (m.name.length * 10 + 34);
  }});

  // Right: Resolutions with hatch pattern swatches
  let rightX = MARGIN.left + PLOT_WIDTH;

  const sizes = [
    {{ s: 672, id: "chkSize672", label: "672×672 (斜线)" }},
    {{ s: 448, id: "chkSize448", label: "448×448 (横线)" }},
    {{ s: 224, id: "chkSize224", label: "224×224 (圆点)" }}
  ];

  sizes.reverse().forEach(sz => {{
    const cb = document.getElementById(sz.id);
    const isChecked = cb ? cb.checked : true;

    rightX -= 124;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "legend-item");
    if (!isChecked) g.setAttribute("opacity", "0.35");
    g.addEventListener("click", () => {{
      if (cb) {{
        cb.checked = !cb.checked;
        updateYBounds();
        render();
      }}
    }});

    // Swatch base
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", rightX);
    rect.setAttribute("y", MARGIN.top - 28);
    rect.setAttribute("width", 18);
    rect.setAttribute("height", 12);
    rect.setAttribute("rx", "2");
    rect.setAttribute("fill", "#475569");
    rect.setAttribute("stroke", "#334155");
    rect.setAttribute("stroke-width", "0.8");
    g.appendChild(rect);

    // Swatch hatch pattern overlay
    const patRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    patRect.setAttribute("x", rightX);
    patRect.setAttribute("y", MARGIN.top - 28);
    patRect.setAttribute("width", 18);
    patRect.setAttribute("height", 12);
    patRect.setAttribute("rx", "2");
    patRect.setAttribute("fill", `url(#pat-${{sz.s}})`);
    g.appendChild(patRect);

    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", rightX + 24);
    txt.setAttribute("y", MARGIN.top - 18);
    txt.setAttribute("class", "axis-text");
    txt.setAttribute("font-weight", "600");
    txt.setAttribute("fill", "#334155");
    txt.textContent = sz.label;
    g.appendChild(txt);

    group.appendChild(g);
  }});
}}

const tooltip = document.getElementById("tooltip");

function showTooltip(e, p) {{
  const activeCfg = METRIC_CONFIG[currentMetric];
  const activeValStr = formatMetricVal(p.y_val, currentMetric);
  const sizeDesc = p.size === 672 ? "672 × 672 (斜线)" : (p.size === 448 ? "448 × 448 (横线)" : "224 × 224 (圆点)");

  tooltip.innerHTML = `
    <div class="tt-title">
      <span style="width: 10px; height: 10px; border-radius: 50%; background: ${{p.color}};"></span>
      <span>${{p.model}}</span>
    </div>
    <div class="tt-row"><span>评测数据集:</span> <span class="tt-val">${{DATASET_NAME}}</span></div>
    <div class="tt-row tt-active-row">
      <span style="font-weight:700; color:#e0f2fe;">${{activeCfg.label}}:</span>
      <span class="tt-val tt-highlight">${{activeValStr}}</span>
    </div>
    <div class="divider" style="margin: 6px 0; opacity: 0.3;"></div>
    <div class="tt-row"><span style="color:#93c5fd; font-weight:700;">【测试集真实评测】</span></div>
    <div class="tt-row"><span>测试集 I-AUROC:</span> <span class="tt-val">${{p.auc.toFixed(4)}}</span></div>
    <div class="tt-row"><span>测试集最优 F1:</span> <span class="tt-val">${{p.f1 ? p.f1.toFixed(4) : "N/A"}}</span></div>
    <div class="tt-row"><span>测试集平均精度 (AP):</span> <span class="tt-val">${{p.ap ? p.ap.toFixed(4) : "N/A"}}</span></div>
    <div class="tt-row"><span>缺陷检出 (TP):</span> <span class="tt-val" style="color:#4ade80; font-weight:600;">${{p.tp}} 张 (检出率: ${{p.recall.toFixed(1)}}%)</span></div>
    <div class="tt-row"><span>缺陷漏检 (FN):</span> <span class="tt-val" style="color:#f43f5e; font-weight:700;">${{p.fn}} 张 (漏检率: ${{p.fnr.toFixed(1)}}%)</span></div>
    <div class="tt-row"><span>测试集良品误报 (FP):</span> <span class="tt-val" style="color:#f87171;">${{p.fp}} 张 (误报率: ${{p.fpr.toFixed(1)}}%)</span></div>
    <div class="tt-row"><span>测试集良品放行 (TN):</span> <span class="tt-val" style="color:#94a3b8;">${{p.tn}} 张</span></div>
    <div class="divider" style="margin: 6px 0; opacity: 0.3;"></div>
    <div class="tt-row"><span style="color:#93c5fd; font-weight:700;">【训练集自检与训练开销】</span></div>
    <div class="tt-row"><span>训练集良品自误报 (Clean FPR):</span> <span class="tt-val" style="color:${{p.clean_fpr > 1 ? '#f87171' : '#4ade80'}}; font-weight:600;">${{p.clean_fpr.toFixed(1)}}%</span></div>
    <div class="tt-row"><span>正常训练样本量:</span> <span class="tt-val" style="color:#fbbf24; font-weight:700;">N = ${{p.n}}</span></div>
    ${{NON_ITERATION_METRICS.includes(currentMetric) ? "" : `<div class="tt-row"><span>训练迭代轮次:</span> <span class="tt-val">${{p.iters.toLocaleString()}} 轮</span></div>`}}
    <div class="tt-row"><span>输入分辨率:</span> <span class="tt-val">${{sizeDesc}}</span></div>
    <div class="tt-row"><span>训练耗时 / 显存峰值:</span> <span class="tt-val" style="color:#38bdf8; font-weight:600;">${{p.train_time_m.toFixed(1)}} min | 训练显存: ${{p.train_vram_gb ? p.train_vram_gb.toFixed(2) : 'N/A'}} GB</span></div>
    <div class="divider" style="margin: 6px 0; opacity: 0.3;"></div>
    <div class="tt-row"><span style="color:#93c5fd; font-weight:700;">【硬件与推理效能】</span></div>
    <div class="tt-row"><span>单图推理耗时 / FPS:</span> <span class="tt-val">${{p.lat_ms.toFixed(1)}} ms (${{Math.round(p.fps)}} FPS)</span></div>
    <div class="tt-row"><span>单图推理显存占用:</span> <span class="tt-val" style="color:#a78bfa; font-weight:600;">${{p.vram_gb.toFixed(2)}} GB</span></div>
  `;
  tooltip.style.display = "block";
  updateTooltipPos(e);
}}

function updateTooltipPos(e) {{
  const tip = document.getElementById("tooltip");
  if (!tip) return;
  const tipW = tip.offsetWidth || 300;
  const tipH = tip.offsetHeight || 180;
  let left = e.clientX + 16;
  let top = e.clientY + 16;

  if (left + tipW > window.innerWidth - 16) left = e.clientX - tipW - 16;
  if (top + tipH > window.innerHeight - 16) top = e.clientY - tipH - 16;
  if (left < 10) left = 10;
  if (top < 10) top = 10;

  tip.style.left = `${{left}}px`;
  tip.style.top = `${{top}}px`;
}}

function hideTooltip() {{
  const tooltip = document.getElementById("tooltip");
  tooltip.style.display = "none";
}}

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
  const inPlotX = p.x >= MARGIN.left - 45 && p.x <= MARGIN.left + PLOT_WIDTH + 30;

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

// ============================================================================
// TAB 2: SEPARATE DATASET DISTRIBUTION CHARTS (TRAIN & TEST)
// ============================================================================

function initDatasetStatsView() {{
  const dsTotalEl = document.getElementById("dsStatDatasetTotal");
  const trainNsEl = document.getElementById("dsStatTrainNs");
  const testTotalEl = document.getElementById("dsStatTestTotal");
  const testNormalEl = document.getElementById("dsStatTestNormal");
  const testDefectEl = document.getElementById("dsStatTestDefect");
  const ratioEl = document.getElementById("dsStatRatio");

  if (!DATASET_STATS || !DATASET_STATS.test_total) {{
    return;
  }}

  if (dsTotalEl && DATASET_STATS.dataset_total) {{
    dsTotalEl.textContent = `${{DATASET_STATS.dataset_total.toLocaleString()}} 张`;
  }}

  const nsStr = (DATASET_STATS.train_ns || []).join(" / ") + " 张";
  if (trainNsEl) trainNsEl.textContent = nsStr;
  if (testTotalEl) testTotalEl.textContent = `${{DATASET_STATS.test_total.toLocaleString()}} 张`;

  const okPct = (DATASET_STATS.test_normal / DATASET_STATS.test_total * 100).toFixed(1);
  const ngPct = (DATASET_STATS.test_defect / DATASET_STATS.test_total * 100).toFixed(1);

  if (testNormalEl) testNormalEl.textContent = `${{DATASET_STATS.test_normal.toLocaleString()}} 张 (${{okPct}}%)`;
  if (testDefectEl) testDefectEl.textContent = `${{DATASET_STATS.test_defect.toLocaleString()}} 张 (${{ngPct}}%)`;

  const ratio = DATASET_STATS.test_normal > 0 ? (DATASET_STATS.test_defect / DATASET_STATS.test_normal).toFixed(2) : "N/A";
  if (ratioEl) ratioEl.textContent = `1 : ${{ratio}} (良品 : 缺陷)`;

  // Populate Subclasses Panel in Sidebar
  const subclasses = DATASET_STATS.subclasses || [];
  const subCountEl = document.getElementById("dsStatSubclassCount");
  const subRowEl = document.getElementById("dsStatSubclassRow");
  const subDivEl = document.getElementById("dsStatSubclassDivider");
  const subPanelGroup = document.getElementById("dsSubclassPanelGroup");
  const subPanelCount = document.getElementById("subclassPanelCount");
  const listContainer = document.getElementById("dsSubclassListContainer");

  if (subclasses.length > 0) {{
    if (subDivEl) subDivEl.style.display = "block";
    if (subRowEl) subRowEl.style.display = "flex";
    if (subCountEl) subCountEl.textContent = `${{subclasses.length}} 个子类`;
    if (subPanelGroup) subPanelGroup.style.display = "flex";
    if (subPanelCount) subPanelCount.textContent = `${{subclasses.length}}`;

    if (listContainer) {{
      listContainer.innerHTML = "";
      subclasses.forEach(sub => {{
        const normPct = sub.total > 0 ? (sub.normal / sub.total * 100).toFixed(1) : "0.0";
        const defPct = sub.total > 0 ? (sub.defect / sub.total * 100).toFixed(1) : "0.0";
        const trainQuotas = (DATASET_STATS.train_subclasses && DATASET_STATS.train_subclasses.subclasses) || [];
        const matchTrainSub = trainQuotas.find(ts => ts.name === sub.name);
        let trainQuotaStr = "";
        if (matchTrainSub && matchTrainSub.counts) {{
          const qParts = Object.entries(matchTrainSub.counts).map(([n, cnt]) => `N=${{n}}: ${{cnt}}`);
          if (qParts.length > 0) {{
            trainQuotaStr = `<div style="font-size: 0.70rem; color: #0d9488; font-weight: 600; margin-top: 2px;">
              训练集配额: ${{qParts.join(" | ")}}
            </div>`;
          }}
        }}

        const testSubList = DATASET_STATS.test_subclasses || [];
        const matchTestSub = testSubList.find(ts => ts.name === sub.name);
        let testQuotaStr = "";
        if (matchTestSub) {{
          testQuotaStr = `<div style="font-size: 0.70rem; color: #1e40af; font-weight: 600; margin-top: 1px;">
            测试集样本: ${{matchTestSub.total.toLocaleString()}} (良品: ${{matchTestSub.normal.toLocaleString()}} | 缺陷: ${{matchTestSub.defect.toLocaleString()}})
          </div>`;
        }}

        const item = document.createElement("div");
        item.className = "subclass-panel-item";
        item.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 700; color: #0f172a; font-family: monospace; font-size: 0.82rem;">${{sub.name}}</span>
            <span style="font-weight: 700; color: #3b82f6;">${{sub.total.toLocaleString()}} 张</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748b; margin-top: 1px;">
            <span style="color: #059669; font-weight: 600;">🟢 良品: ${{sub.normal.toLocaleString()}} (${{normPct}}%)</span>
            <span style="color: #dc2626; font-weight: 600;">🔴 缺陷: ${{sub.defect.toLocaleString()}} (${{defPct}}%)</span>
          </div>
          <div style="width: 100%; height: 4px; background: #fee2e2; border-radius: 2px; overflow: hidden; display: flex; margin-top: 3px;">
            <div style="width: ${{normPct}}%; height: 100%; background: #10b981;"></div>
            <div style="width: ${{defPct}}%; height: 100%; background: #f43f5e;"></div>
          </div>
          ${{trainQuotaStr}}
          ${{testQuotaStr}}
        `;
        listContainer.appendChild(item);
      }});
    }}
  }} else {{
    if (subDivEl) subDivEl.style.display = "none";
    if (subRowEl) subRowEl.style.display = "none";
    if (subPanelGroup) subPanelGroup.style.display = "none";
  }}
}}

function generateSubYTicks(minVal, maxVal, maxTicks = 5) {{
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
    ticks.push(Math.round(v));
  }}
  if (ticks.length === 0) ticks.push(minVal, maxVal);
  return ticks;
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
  const yTicks = generateSubYTicks(0, trainYMax, 5);
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

  // X Axis Title
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", SUB_MARGIN.left + SUB_PLOT_WIDTH / 2);
  xTitle.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 46);
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
    valText.textContent = `${{nVal}}`;
    barsGroup.appendChild(valText);

    // Mouse guideline events
    rect.addEventListener("mouseenter", (e) => {{
      updateTrainCursorGuideline(e);
    }});
    rect.addEventListener("mousemove", (e) => {{
      updateTrainCursorGuideline(e);
    }});
    rect.addEventListener("mouseleave", () => {{
      hideTrainCursorGuideline();
    }});
  }});
}}

// 2. Render Test Set Chart
let testYMax = 1000;
function renderTestChart() {{
  const gridGroup = document.getElementById("test-grid-group");
  const axesGroup = document.getElementById("test-axes-group");
  const barsGroup = document.getElementById("test-bars-group");
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
  const yTicks = generateSubYTicks(0, testYMax, 5);
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
  xTitle.setAttribute("y", SUB_MARGIN.top + SUB_PLOT_HEIGHT + 46);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = "测试集样本类别划分";
  axesGroup.appendChild(xTitle);

  // 3 Test Categories
  const testCats = [
    {{ label: "测试良品 (OK)", count: normal, color: "#10b981", desc: "正常无缺陷测试样本" }},
    {{ label: "测试缺陷 (NG)", count: defect, color: "#f43f5e", desc: "真实异常缺陷测试样本" }},
    {{ label: "测试集总量", count: total, color: "#3b82f6", desc: "全量独立验证集" }}
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
    valText.textContent = `${{cat.count.toLocaleString()}}`;
    barsGroup.appendChild(valText);

    // Mouse guideline events
    rect.addEventListener("mouseenter", (e) => {{
      updateTestCursorGuideline(e);
    }});
    rect.addEventListener("mousemove", (e) => {{
      updateTestCursorGuideline(e);
    }});
    rect.addEventListener("mouseleave", () => {{
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
    const valStr = `${{curVal}}`;

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
    const valStr = `${{curVal}}`;

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

let trainSubclassYMax = 50;
let testSubclassYMax = 100;

// Toggle between Overall distribution, Full Subclasses, and Training Subclasses
function setDsViewMode(mode) {{
  currentDsViewMode = mode;
  const selTop = document.getElementById("dsViewSelect");
  if (selTop && selTop.value !== mode) selTop.value = mode;

  const dsGrid = document.getElementById("dsChartsGrid");
  const subBody = document.getElementById("subclassChartBody");
  const trainSubBody = document.getElementById("trainSubclassChartBody");
  const testSubBody = document.getElementById("testSubclassChartBody");
  const titleEl = document.getElementById("dsChartTitle");
  const statsEl = document.getElementById("dsChartStats");
  const actionsEl = document.getElementById("dsChartActions");

  const subCount = (DATASET_STATS.subclasses || []).length;
  const trainNs = (DATASET_STATS.train_subclasses && DATASET_STATS.train_subclasses.ns) || DATASET_STATS.train_ns || [];
  const testSubCount = (DATASET_STATS.test_subclasses || []).length;

  if (mode === "test_subclasses") {{
    if (dsGrid) dsGrid.style.display = "none";
    if (subBody) subBody.style.display = "none";
    if (trainSubBody) trainSubBody.style.display = "none";
    if (testSubBody) testSubBody.style.display = "flex";

    if (titleEl) titleEl.textContent = `📦 ${{DATASET_NAME}} · 测试集各子类样本分布 (Test Subclasses)`;
    if (statsEl) statsEl.textContent = `展示测试集中各子类良品 (OK) 与缺陷 (NG) 样本测试规模及构成 (共 ${{testSubCount}} 个子类)`;

    if (actionsEl) {{
      actionsEl.innerHTML = `
        <div style="display: flex; align-items: center; gap: 14px; font-size: 0.78rem;">
          <span style="display: inline-flex; align-items: center; gap: 5px;">
            <span style="width: 10px; height: 10px; background: #059669; border-radius: 2px;"></span> 测试良品 (OK)
          </span>
          <span style="display: inline-flex; align-items: center; gap: 5px;">
            <span style="width: 10px; height: 10px; background: #dc2626; border-radius: 2px;"></span> 测试缺陷 (NG)
          </span>
        </div>
      `;
    }}
    renderTestSubclassChart();

  }} else if (mode === "train_subclasses") {{
    if (dsGrid) dsGrid.style.display = "none";
    if (subBody) subBody.style.display = "none";
    if (trainSubBody) trainSubBody.style.display = "flex";
    if (testSubBody) testSubBody.style.display = "none";

    if (titleEl) titleEl.textContent = `📦 ${{DATASET_NAME}} · 训练集各子类样本分布 (Training Subclasses)`;
    if (statsEl) statsEl.textContent = `展示各采样阶梯规模 N 下，训练集中各子类良品样本的实际抽取数量 (共 ${{subCount}} 个子类)`;

    if (actionsEl) {{
      const colors = ["#0d9488", "#0ea5e9", "#3b82f6", "#6366f1", "#8b5cf6"];
      let legendHtml = `<div style="display: flex; align-items: center; gap: 14px; font-size: 0.78rem;">`;
      trainNs.forEach((nVal, i) => {{
        const c = colors[i % colors.length];
        legendHtml += `<span style="display: inline-flex; align-items: center; gap: 5px;">
          <span style="width: 10px; height: 10px; background: ${{c}}; border-radius: 2px;"></span> N = ${{nVal}}
        </span>`;
      }});
      legendHtml += `</div>`;
      actionsEl.innerHTML = legendHtml;
    }}
    renderTrainSubclassChart();

  }} else if (mode === "subclasses") {{
    if (dsGrid) dsGrid.style.display = "none";
    if (subBody) subBody.style.display = "flex";
    if (trainSubBody) trainSubBody.style.display = "none";
    if (testSubBody) testSubBody.style.display = "none";

    if (titleEl) titleEl.textContent = `📦 ${{DATASET_NAME}} · 全量各子类样本分布 (Subclasses)`;
    if (statsEl) statsEl.textContent = `展示真实文件目录扫描得到的各子类样本数量及良品/缺陷构成 (共 ${{subCount}} 个子类)`;

    if (actionsEl) {{
      actionsEl.innerHTML = `
        <div style="display: flex; align-items: center; gap: 14px; font-size: 0.78rem;">
          <span style="display: inline-flex; align-items: center; gap: 5px;">
            <span style="width: 10px; height: 10px; background: #10b981; border-radius: 2px;"></span> 正常良品
          </span>
          <span style="display: inline-flex; align-items: center; gap: 5px;">
            <span style="width: 10px; height: 10px; background: #f43f5e; border-radius: 2px;"></span> 缺陷异常
          </span>
        </div>
      `;
    }}
    renderSubclassChart();

  }} else {{
    // "overall"
    if (dsGrid) dsGrid.style.display = "flex";
    if (subBody) subBody.style.display = "none";
    if (trainSubBody) trainSubBody.style.display = "none";
    if (testSubBody) testSubBody.style.display = "none";

    if (titleEl) titleEl.textContent = `📦 ${{DATASET_NAME}} · 样本量分布看板 (训练集 vs 测试集)`;
    if (statsEl) statsEl.textContent = `展示当前数据集自身的训练规模与测试集构成`;
    if (actionsEl) actionsEl.innerHTML = "";

    renderTrainChart();
    renderTestChart();
  }}
}}

// 3. Render Subclass Chart (Scanned from real directory)
function renderSubclassChart() {{
  const subclasses = DATASET_STATS.subclasses || [];
  const body = document.getElementById("subclassChartBody");
  if (!body) return;
  if (!subclasses || subclasses.length === 0) {{
    body.style.display = "none";
    return;
  }}
  body.style.display = "flex";

  const gridGroup = document.getElementById("subclass-grid-group");
  const axesGroup = document.getElementById("subclass-axes-group");
  const barsGroup = document.getElementById("subclass-bars-group");
  if (!gridGroup || !axesGroup || !barsGroup) return;

  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  barsGroup.innerHTML = "";

  const maxVal = Math.max(...subclasses.map(s => Math.max(s.normal, s.defect)), 10);
  subclassYMax = Math.ceil(maxVal * 1.25 / 50) * 50;

  function getY(val) {{
    const ratio = Math.max(0, Math.min(1, val / subclassYMax));
    return SUBCLASS_MARGIN.top + (1.0 - ratio) * SUBCLASS_PLOT_HEIGHT;
  }}

  // Y Ticks
  const yTicks = generateSubYTicks(0, subclassYMax, 5);
  yTicks.forEach(yVal => {{
    const py = getY(yVal);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", SUBCLASS_MARGIN.left);
    line.setAttribute("y1", py);
    line.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
    line.setAttribute("y2", py);
    line.setAttribute("class", "grid-line");
    gridGroup.appendChild(line);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", SUBCLASS_MARGIN.left - 8);
    text.setAttribute("y", py + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("class", "axis-text");
    text.textContent = yVal.toLocaleString();
    axesGroup.appendChild(text);
  }});

  // Y Spine & Title
  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", SUBCLASS_MARGIN.left);
  ySpine.setAttribute("y1", SUBCLASS_MARGIN.top);
  ySpine.setAttribute("x2", SUBCLASS_MARGIN.left);
  ySpine.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("x", -(SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", SUBCLASS_MARGIN.left - 44);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = "样本数量 (张)";
  axesGroup.appendChild(yTitle);

  // X Spine
  const xSpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xSpine.setAttribute("x1", SUBCLASS_MARGIN.left);
  xSpine.setAttribute("y1", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  xSpine.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
  xSpine.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  xSpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(xSpine);

  // X Title
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH / 2);
  xTitle.setAttribute("y", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 46);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = "数据集各子类目录 (Subdirectories / Subclasses)";
  axesGroup.appendChild(xTitle);

  const slotW = SUBCLASS_PLOT_WIDTH / Math.max(1, subclasses.length);
  const barW = Math.min(26, Math.max(10, slotW * 0.32));
  const barGap = Math.min(6, Math.max(2, slotW * 0.05));

  subclasses.forEach((sub, idx) => {{
    const cx = SUBCLASS_MARGIN.left + (idx + 0.5) * slotW;

    // Tick line
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", cx);
    tick.setAttribute("y1", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
    tick.setAttribute("x2", cx);
    tick.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 5);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    // X Label
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", cx);
    text.setAttribute("y", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 22);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "axis-text");
    text.setAttribute("font-weight", "600");
    text.textContent = sub.name;
    axesGroup.appendChild(text);

    // Group center offset
    const leftX = cx - barW - barGap / 2;
    const rightX = cx + barGap / 2;

    // 1. Normal Bar (Green)
    const normPy = getY(sub.normal);
    const normH = (SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT) - normPy;
    const normRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    normRect.setAttribute("x", leftX);
    normRect.setAttribute("y", normPy);
    normRect.setAttribute("width", barW);
    normRect.setAttribute("height", normH);
    normRect.setAttribute("fill", "#10b981");
    normRect.setAttribute("rx", "3");
    normRect.setAttribute("class", "bar-rect");
    barsGroup.appendChild(normRect);

    if (normH > 14 || sub.normal > 0) {{
      const normVal = document.createElementNS("http://www.w3.org/2000/svg", "text");
      normVal.setAttribute("x", leftX + barW / 2);
      normVal.setAttribute("y", normPy - 5);
      normVal.setAttribute("text-anchor", "middle");
      normVal.setAttribute("class", "val-label");
      normVal.setAttribute("fill", "#059669");
      normVal.setAttribute("font-size", "10");
      normVal.textContent = sub.normal.toLocaleString();
      barsGroup.appendChild(normVal);
    }}

    // 2. Defect Bar (Red)
    const defPy = getY(sub.defect);
    const defH = (SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT) - defPy;
    const defRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    defRect.setAttribute("x", rightX);
    defRect.setAttribute("y", defPy);
    defRect.setAttribute("width", barW);
    defRect.setAttribute("height", defH);
    defRect.setAttribute("fill", "#f43f5e");
    defRect.setAttribute("rx", "3");
    defRect.setAttribute("class", "bar-rect");
    barsGroup.appendChild(defRect);

    if (defH > 14 || sub.defect > 0) {{
      const defVal = document.createElementNS("http://www.w3.org/2000/svg", "text");
      defVal.setAttribute("x", rightX + barW / 2);
      defVal.setAttribute("y", defPy - 5);
      defVal.setAttribute("text-anchor", "middle");
      defVal.setAttribute("class", "val-label");
      defVal.setAttribute("fill", "#dc2626");
      defVal.setAttribute("font-size", "10");
      defVal.textContent = sub.defect.toLocaleString();
      barsGroup.appendChild(defVal);
    }}

    // Mouse guideline events
    const attachHover = (el) => {{
      el.addEventListener("mouseenter", (e) => updateSubclassCursorGuideline(e));
      el.addEventListener("mousemove", (e) => updateSubclassCursorGuideline(e));
      el.addEventListener("mouseleave", () => hideSubclassCursorGuideline());
    }};

    attachHover(normRect);
    attachHover(defRect);
  }});
}}

function updateSubclassCursorGuideline(e) {{
  const cursorGroup = document.getElementById("subclass-cursor-group");
  const cursorHLine = document.getElementById("subclassCursorHLine");
  const cursorBadge = document.getElementById("subclassCursorBadge");
  const cursorBadgeRect = document.getElementById("subclassCursorBadgeRect");
  const cursorBadgeText = document.getElementById("subclassCursorBadgeText");
  if (!cursorGroup || !cursorHLine) return;

  const p = getSvgCoords(e, "subclass-chart", SUBCLASS_SVG_WIDTH, SUBCLASS_SVG_HEIGHT);
  if (!p) return;

  const inPlotY = p.y >= SUBCLASS_MARGIN.top && p.y <= SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT;
  const inPlotX = p.x >= SUBCLASS_MARGIN.left - 30 && p.x <= SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH + 20;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";
    cursorHLine.setAttribute("x1", SUBCLASS_MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    const ratio = 1.0 - (p.y - SUBCLASS_MARGIN.top) / SUBCLASS_PLOT_HEIGHT;
    const curVal = Math.round(ratio * subclassYMax);
    const valStr = `${{curVal}}`;

    if (cursorBadge && cursorBadgeText && cursorBadgeRect) {{
      cursorBadge.style.display = "block";
      cursorBadgeText.textContent = valStr;
      const badgeW = Math.max(50, valStr.length * 7.5 + 12);
      const badgeH = 19;
      const badgeX = Math.max(2, SUBCLASS_MARGIN.left - badgeW - 5);
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

function hideSubclassCursorGuideline() {{
  const g = document.getElementById("subclass-cursor-group");
  if (g) g.style.display = "none";
}}

// 4. Render Training Set Subclasses Chart
function renderTrainSubclassChart() {{
  const trainData = DATASET_STATS.train_subclasses || {{}};
  const ns = trainData.ns || [];
  const subclasses = trainData.subclasses || [];
  const body = document.getElementById("trainSubclassChartBody");
  if (!body) return;

  if (!subclasses || subclasses.length === 0 || !ns || ns.length === 0) {{
    body.style.display = "none";
    return;
  }}
  body.style.display = "flex";

  const gridGroup = document.getElementById("train-subclass-grid-group");
  const axesGroup = document.getElementById("train-subclass-axes-group");
  const barsGroup = document.getElementById("train-subclass-bars-group");
  if (!gridGroup || !axesGroup || !barsGroup) return;

  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  barsGroup.innerHTML = "";

  // Compute max value
  let maxVal = 10;
  subclasses.forEach(s => {{
    Object.values(s.counts || {{}}).forEach(v => {{
      if (v > maxVal) maxVal = v;
    }});
  }});
  trainSubclassYMax = Math.ceil(maxVal * 1.25 / 10) * 10 || 50;

  function getY(val) {{
    const ratio = Math.max(0, Math.min(1, val / trainSubclassYMax));
    return SUBCLASS_MARGIN.top + (1.0 - ratio) * SUBCLASS_PLOT_HEIGHT;
  }}

  // Y Ticks
  const yTicks = generateSubYTicks(0, trainSubclassYMax, 5);
  yTicks.forEach(yVal => {{
    const py = getY(yVal);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", SUBCLASS_MARGIN.left);
    line.setAttribute("y1", py);
    line.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
    line.setAttribute("y2", py);
    line.setAttribute("class", "grid-line");
    gridGroup.appendChild(line);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", SUBCLASS_MARGIN.left - 8);
    text.setAttribute("y", py + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("class", "axis-text");
    text.textContent = yVal.toLocaleString();
    axesGroup.appendChild(text);
  }});

  // Y Spine & Title
  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", SUBCLASS_MARGIN.left);
  ySpine.setAttribute("y1", SUBCLASS_MARGIN.top);
  ySpine.setAttribute("x2", SUBCLASS_MARGIN.left);
  ySpine.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("x", -(SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", SUBCLASS_MARGIN.left - 44);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = "训练良品样本数量";
  axesGroup.appendChild(yTitle);

  // X Spine
  const xSpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xSpine.setAttribute("x1", SUBCLASS_MARGIN.left);
  xSpine.setAttribute("y1", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  xSpine.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
  xSpine.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  xSpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(xSpine);

  // X Title
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH / 2);
  xTitle.setAttribute("y", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 46);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = "训练集各子类目录 (Training Subclasses)";
  axesGroup.appendChild(xTitle);

  const TRAIN_COLORS = ["#0d9488", "#0ea5e9", "#3b82f6", "#6366f1", "#8b5cf6"];
  const numTiers = ns.length;
  const slotW = SUBCLASS_PLOT_WIDTH / Math.max(1, subclasses.length);
  const clusterW = Math.min(slotW * 0.70, numTiers * 28);
  const barW = Math.max(8, (clusterW - (numTiers - 1) * 3) / numTiers);
  const barGap = 3;

  subclasses.forEach((sub, sIdx) => {{
    const cx = SUBCLASS_MARGIN.left + (sIdx + 0.5) * slotW;

    // Tick line
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", cx);
    tick.setAttribute("y1", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
    tick.setAttribute("x2", cx);
    tick.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 5);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    // X Label
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", cx);
    text.setAttribute("y", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 22);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "axis-text");
    text.setAttribute("font-weight", "600");
    text.textContent = sub.name;
    axesGroup.appendChild(text);

    // Cluster start X
    const startX = cx - (numTiers * barW + (numTiers - 1) * barGap) / 2;

    ns.forEach((nVal, tIdx) => {{
      const cnt = (sub.counts && sub.counts[String(nVal)]) || 0;
      const bx = startX + tIdx * (barW + barGap);
      const py = getY(cnt);
      const bHeight = (SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT) - py;
      const bColor = TRAIN_COLORS[tIdx % TRAIN_COLORS.length];

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", bx);
      rect.setAttribute("y", py);
      rect.setAttribute("width", barW);
      rect.setAttribute("height", bHeight);
      rect.setAttribute("fill", bColor);
      rect.setAttribute("rx", "3");
      rect.setAttribute("class", "bar-rect");
      barsGroup.appendChild(rect);

      // Value label on top (without unit '张', as requested)
      if (bHeight > 12 || cnt > 0) {{
        const valText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        valText.setAttribute("x", bx + barW / 2);
        valText.setAttribute("y", py - 5);
        valText.setAttribute("text-anchor", "middle");
        valText.setAttribute("class", "val-label");
        valText.setAttribute("fill", bColor);
        valText.setAttribute("font-size", numTiers > 3 ? "9" : "10");
        valText.textContent = `${{cnt}}`;
        barsGroup.appendChild(valText);
      }}

      rect.addEventListener("mouseenter", (e) => updateTrainSubclassCursorGuideline(e));
      rect.addEventListener("mousemove", (e) => updateTrainSubclassCursorGuideline(e));
      rect.addEventListener("mouseleave", () => hideTrainSubclassCursorGuideline());
    }});
  }});
}}

function updateTrainSubclassCursorGuideline(e) {{
  const cursorGroup = document.getElementById("train-subclass-cursor-group");
  const cursorHLine = document.getElementById("trainSubclassCursorHLine");
  const cursorBadge = document.getElementById("trainSubclassCursorBadge");
  const cursorBadgeRect = document.getElementById("trainSubclassCursorBadgeRect");
  const cursorBadgeText = document.getElementById("trainSubclassCursorBadgeText");
  if (!cursorGroup || !cursorHLine) return;

  const p = getSvgCoords(e, "train-subclass-chart", SUBCLASS_SVG_WIDTH, SUBCLASS_SVG_HEIGHT);
  if (!p) return;

  const inPlotY = p.y >= SUBCLASS_MARGIN.top && p.y <= SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT;
  const inPlotX = p.x >= SUBCLASS_MARGIN.left - 30 && p.x <= SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH + 20;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";
    cursorHLine.setAttribute("x1", SUBCLASS_MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    const ratio = 1.0 - (p.y - SUBCLASS_MARGIN.top) / SUBCLASS_PLOT_HEIGHT;
    const curVal = Math.round(ratio * trainSubclassYMax);
    const valStr = `${{curVal}}`;

    if (cursorBadge && cursorBadgeText && cursorBadgeRect) {{
      cursorBadge.style.display = "block";
      cursorBadgeText.textContent = valStr;
      const badgeW = Math.max(46, valStr.length * 8 + 12);
      const badgeH = 19;
      const badgeX = Math.max(2, SUBCLASS_MARGIN.left - badgeW - 5);
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

function hideTrainSubclassCursorGuideline() {{
  const g = document.getElementById("train-subclass-cursor-group");
  if (g) g.style.display = "none";
}}

// 5. Render Test Subclass Chart (from test_full.txt)
function renderTestSubclassChart() {{
  const testSubclasses = DATASET_STATS.test_subclasses || [];
  const body = document.getElementById("testSubclassChartBody");
  if (!body) return;
  if (!testSubclasses || testSubclasses.length === 0) {{
    body.style.display = "none";
    return;
  }}

  const gridGroup = document.getElementById("test-subclass-grid-group");
  const axesGroup = document.getElementById("test-subclass-axes-group");
  const barsGroup = document.getElementById("test-subclass-bars-group");

  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  barsGroup.innerHTML = "";

  const maxVal = Math.max(...testSubclasses.map(s => Math.max(s.normal, s.defect)), 10);
  const niceMax = Math.ceil(maxVal * 1.18);
  testSubclassYMax = niceMax;

  const yTicks = generateSubYTicks(0, niceMax, 5);

  // Grid lines & Y Axis Ticks
  yTicks.forEach(yVal => {{
    const py = SUBCLASS_MARGIN.top + (1.0 - yVal / niceMax) * SUBCLASS_PLOT_HEIGHT;

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", SUBCLASS_MARGIN.left);
    line.setAttribute("y1", py);
    line.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
    line.setAttribute("y2", py);
    line.setAttribute("class", "grid-line");
    gridGroup.appendChild(line);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", SUBCLASS_MARGIN.left - 8);
    text.setAttribute("y", py + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("class", "axis-text");
    text.textContent = yVal.toLocaleString();
    axesGroup.appendChild(text);
  }});

  // Y Spine & Title
  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", SUBCLASS_MARGIN.left);
  ySpine.setAttribute("y1", SUBCLASS_MARGIN.top);
  ySpine.setAttribute("x2", SUBCLASS_MARGIN.left);
  ySpine.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("x", -(SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", SUBCLASS_MARGIN.left - 44);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = "测试样本数量";
  axesGroup.appendChild(yTitle);

  // X Spine
  const xSpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xSpine.setAttribute("x1", SUBCLASS_MARGIN.left);
  xSpine.setAttribute("y1", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  xSpine.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
  xSpine.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
  xSpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(xSpine);

  // X Title
  const xTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xTitle.setAttribute("x", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH / 2);
  xTitle.setAttribute("y", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 46);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-title");
  xTitle.textContent = "测试集各子类 (Test Subclasses)";
  axesGroup.appendChild(xTitle);

  const slotW = SUBCLASS_PLOT_WIDTH / Math.max(1, testSubclasses.length);
  const barW = Math.min(26, Math.max(10, slotW * 0.32));
  const barGap = Math.min(6, Math.max(2, slotW * 0.05));

  testSubclasses.forEach((sub, idx) => {{
    const cx = SUBCLASS_MARGIN.left + (idx + 0.5) * slotW;

    // Tick line
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", cx);
    tick.setAttribute("y1", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT);
    tick.setAttribute("x2", cx);
    tick.setAttribute("y2", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 5);
    tick.setAttribute("class", "axis-tick");
    axesGroup.appendChild(tick);

    // X Label
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", cx);
    text.setAttribute("y", SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT + 22);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "axis-text");
    text.setAttribute("font-weight", "600");
    text.textContent = sub.name;
    axesGroup.appendChild(text);

    // Group center offset
    const leftX = cx - barW - barGap / 2;
    const rightX = cx + barGap / 2;

    const bars = [
      {{ x: leftX, val: sub.normal, color: "#059669", label: "测试良品 (OK)" }},
      {{ x: rightX, val: sub.defect, color: "#dc2626", label: "测试缺陷 (NG)" }}
    ];

    bars.forEach(b => {{
      const bh = (b.val / niceMax) * SUBCLASS_PLOT_HEIGHT;
      const by = SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT - bh;

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", b.x);
      rect.setAttribute("y", by);
      rect.setAttribute("width", barW);
      rect.setAttribute("height", Math.max(1, bh));
      rect.setAttribute("fill", b.color);
      rect.setAttribute("rx", "3");
      rect.setAttribute("ry", "3");
      rect.style.cursor = "default";
      barsGroup.appendChild(rect);

      // Value label on top (without unit '张')
      if (b.val > 0) {{
        const valText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        valText.setAttribute("x", b.x + barW / 2);
        valText.setAttribute("y", by - 6);
        valText.setAttribute("text-anchor", "middle");
        valText.setAttribute("class", "axis-text");
        valText.setAttribute("font-weight", "700");
        valText.setAttribute("font-size", testSubclasses.length > 10 ? "10" : "11");
        valText.setAttribute("fill", b.color);
        valText.textContent = `${{b.val}}`;
        barsGroup.appendChild(valText);
      }}
    }});
  }});
}}

function updateTestSubclassCursorGuideline(e) {{
  const cursorGroup = document.getElementById("test-subclass-cursor-group");
  const cursorHLine = document.getElementById("testSubclassCursorHLine");
  const cursorBadge = document.getElementById("testSubclassCursorBadge");
  const cursorBadgeRect = document.getElementById("testSubclassCursorBadgeRect");
  const cursorBadgeText = document.getElementById("testSubclassCursorBadgeText");
  if (!cursorGroup || !cursorHLine) return;

  const p = getSvgCoords(e, "test-subclass-chart", 1300, 500);
  if (!p) return;

  const inPlotY = p.y >= SUBCLASS_MARGIN.top && p.y <= SUBCLASS_MARGIN.top + SUBCLASS_PLOT_HEIGHT;
  const inPlotX = p.x >= SUBCLASS_MARGIN.left - 30 && p.x <= SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH + 20;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";
    cursorHLine.setAttribute("x1", SUBCLASS_MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", SUBCLASS_MARGIN.left + SUBCLASS_PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    const ratio = 1.0 - (p.y - SUBCLASS_MARGIN.top) / SUBCLASS_PLOT_HEIGHT;
    const curVal = Math.round(ratio * testSubclassYMax);
    const valStr = `${{curVal}}`;

    if (cursorBadge && cursorBadgeText && cursorBadgeRect) {{
      cursorBadge.style.display = "block";
      cursorBadgeText.textContent = valStr;
      const badgeW = Math.max(46, valStr.length * 8 + 12);
      const badgeH = 19;
      const badgeX = Math.max(2, SUBCLASS_MARGIN.left - badgeW - 5);
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

function hideTestSubclassCursorGuideline() {{
  const g = document.getElementById("test-subclass-cursor-group");
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
  const dsSelectors = document.getElementById("datasetSelectors");
  const btnExport = document.getElementById("btnExport");

  if (tab === "benchmark") {{
    tabBench.classList.add("active");
    tabDs.classList.remove("active");
    benchArea.style.display = "flex";
    dsArea.style.display = "none";
    benchControls.style.display = "flex";
    dsControls.style.display = "none";
    if (benchSelectors) benchSelectors.style.display = "flex";
    if (dsSelectors) dsSelectors.style.display = "none";
    if (btnExport) {{
      btnExport.style.display = "inline-flex";
      btnExport.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 导出`;
    }}
    render();
  }} else {{
    tabBench.classList.remove("active");
    tabDs.classList.add("active");
    benchArea.style.display = "none";
    dsArea.style.display = "flex";
    benchControls.style.display = "none";
    dsControls.style.display = "flex";
    if (benchSelectors) benchSelectors.style.display = "none";
    if (dsSelectors) dsSelectors.style.display = "flex";
    if (btnExport) btnExport.style.display = "none";
    hideTooltip();
    setDsViewMode(currentDsViewMode);
  }}
}}

function showDetailCard(p) {{
  const panel = document.getElementById("detailPanel");
  if (!panel) return;
  const activeCfg = METRIC_CONFIG[currentMetric];
  const sizeDesc = p.size === 672 ? "672 × 672 (斜线)" : (p.size === 448 ? "448 × 448 (横线)" : "224 × 224 (圆点)");
  panel.innerHTML = `
    <span><strong>选中实验:</strong> <span style="color: ${{p.color}}; font-weight: 700;">${{p.model}}</span> | <strong>${{activeCfg.label}}:</strong> <span style="color: #2563eb; font-weight: 700;">${{formatMetricVal(p.y_val, currentMetric)}}</span> | <strong>测试集AUROC:</strong> ${{p.auc.toFixed(4)}} | <strong>最优F1:</strong> ${{p.f1 ? p.f1.toFixed(4) : 'N/A'}} | <strong>测试集检出(TP)/漏检(FN)/误报(FP):</strong> <span style="color:#16a34a; font-weight:600;">${{p.tp}}</span> (${{p.recall.toFixed(1)}}%) / <span style="color:#dc2626; font-weight:700;">${{p.fn}}</span> (漏检: ${{p.fnr.toFixed(1)}}%) / <span style="color:#ea580c; font-weight:600;">${{p.fp}}</span> (误报: ${{p.fpr.toFixed(1)}}%) | <strong>训练集良品自误报:</strong> ${{p.clean_fpr.toFixed(1)}}% | <strong>N:</strong> ${{p.n}} ${{NON_ITERATION_METRICS.includes(currentMetric) ? "" : `| <strong>迭代:</strong> ${{p.iters.toLocaleString()}}`}} | <strong>尺寸:</strong> ${{sizeDesc}} | <strong>训练耗时:</strong> ${{p.train_time_m.toFixed(1)}}min | <strong>训练显存:</strong> ${{p.train_vram_gb ? p.train_vram_gb.toFixed(2) : 'N/A'}}GB | <strong>推理延迟:</strong> ${{p.lat_ms.toFixed(1)}}ms | <strong>推理显存:</strong> ${{p.vram_gb.toFixed(2)}}GB</span>
  `;
}}

function setupEventListeners() {{
  const chartBody = document.getElementById("chartBody");

  chartBody.addEventListener("mousedown", (e) => {{
    if (e.target.classList.contains("bar-rect") || (e.target.closest && e.target.closest(".bar-item"))) return;
    isDragging = true;
    startY = e.clientY;
    dragStartYMin = currentYMin;
    dragStartYMax = currentYMax;
  }});

  window.addEventListener("mousemove", (e) => {{
    if (!isDragging) return;
    const dy = e.clientY - startY;
    const ySpan = dragStartYMax - dragStartYMin;
    const yShift = (dy / PLOT_HEIGHT) * ySpan;
    currentYMin = dragStartYMin + yShift;
    currentYMax = dragStartYMax + yShift;
    render();
  }});

  window.addEventListener("mouseup", () => {{
    isDragging = false;
  }});

  chartBody.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const rect = chartBody.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    zoomAt(zoomFactor, mouseY);
  }}, {{ passive: false }});

  chartBody.addEventListener("mousemove", (e) => {{
    updateCursorGuideline(e);
  }});

  chartBody.addEventListener("mouseleave", () => {{
    hideCursorGuideline();
    hideTooltip();
  }});

  const trainBody = document.getElementById("trainChartBody");
  if (trainBody) {{
    trainBody.addEventListener("mousemove", (e) => updateTrainCursorGuideline(e));
    trainBody.addEventListener("mouseleave", () => hideTrainCursorGuideline());
  }}

  const testBody = document.getElementById("testChartBody");
  if (testBody) {{
    testBody.addEventListener("mousemove", (e) => updateTestCursorGuideline(e));
    testBody.addEventListener("mouseleave", () => hideTestCursorGuideline());
  }}

  const subclassBody = document.getElementById("subclassChartBody");
  if (subclassBody) {{
    subclassBody.addEventListener("mousemove", (e) => updateSubclassCursorGuideline(e));
    subclassBody.addEventListener("mouseleave", () => {{
      hideSubclassCursorGuideline();
      hideTooltip();
    }});
  }}

  const trainSubclassBody = document.getElementById("trainSubclassChartBody");
  if (trainSubclassBody) {{
    trainSubclassBody.addEventListener("mousemove", (e) => updateTrainSubclassCursorGuideline(e));
    trainSubclassBody.addEventListener("mouseleave", () => {{
      hideTrainSubclassCursorGuideline();
      hideTooltip();
    }});
  }}

  const testSubclassBody = document.getElementById("testSubclassChartBody");
  if (testSubclassBody) {{
    testSubclassBody.addEventListener("mousemove", (e) => updateTestSubclassCursorGuideline(e));
    testSubclassBody.addEventListener("mouseleave", () => {{
      hideTestSubclassCursorGuideline();
      hideTooltip();
    }});
  }}

  const dsSel = document.getElementById("dsViewSelect");
  if (dsSel) {{
    dsSel.addEventListener("change", (e) => setDsViewMode(e.target.value));
  }}

  document.getElementById("btnZoomIn").addEventListener("click", () => zoomAt(1.25, MARGIN.top + PLOT_HEIGHT / 2));
  document.getElementById("btnZoomOut").addEventListener("click", () => zoomAt(0.8, MARGIN.top + PLOT_HEIGHT / 2));
  document.getElementById("btnResetView").addEventListener("click", resetView);

  document.getElementById("metricSelector").addEventListener("change", (e) => {{
    currentMetric = e.target.value;
    if (NON_ITERATION_METRICS.includes(currentMetric)) {{
      currentXDim = "n";
    }}
    updateXAxisSelectorOptions();
    updateYBounds();
    resetView();
  }});

  document.getElementById("xAxisSelector").addEventListener("change", (e) => {{
    currentXDim = e.target.value;
    if (currentXDim === "iters" && NON_ITERATION_METRICS.includes(currentMetric)) {{
      currentMetric = "auc";
      document.getElementById("metricSelector").value = "auc";
      updateXAxisSelectorOptions();
    }}
    updateYBounds();
    resetView();
  }});

  const modelControlIds = HAS_BANK ? ["chkE2E", "chkDino", "chkPatch"] : ["chkDino", "chkPatch"];
  [...modelControlIds, "chkSize224", "chkSize448", "chkSize672"].forEach(id => {{
    const el = document.getElementById(id);
    if (el) {{
      el.addEventListener("change", () => {{
        updateYBounds();
        render();
      }});
    }}
  }});

  document.getElementById("yAxisRangeMode").addEventListener("change", () => {{
    updateYBounds();
    render();
  }});

  document.getElementById("btnSelectAllModels").addEventListener("click", () => {{
    modelControlIds.forEach(id => {{
      const el = document.getElementById(id);
      if (el) el.checked = true;
    }});
    updateYBounds();
    render();
  }});
  document.getElementById("btnUnselectAllModels").addEventListener("click", () => {{
    modelControlIds.forEach(id => {{
      const el = document.getElementById(id);
      if (el) el.checked = false;
    }});
    updateYBounds();
    render();
  }});

  document.getElementById("btnSelectAllSizes").addEventListener("click", () => {{
    ["chkSize224", "chkSize448", "chkSize672"].forEach(id => document.getElementById(id).checked = true);
    updateYBounds();
    render();
  }});
  document.getElementById("btnUnselectAllSizes").addEventListener("click", () => {{
    ["chkSize224", "chkSize448", "chkSize672"].forEach(id => document.getElementById(id).checked = false);
    updateYBounds();
    render();
  }});

  document.getElementById("btnSelectAllN").addEventListener("click", () => {{
    document.querySelectorAll("[data-n]").forEach(cb => cb.checked = true);
    updateYBounds();
    render();
  }});
  document.getElementById("btnUnselectAllN").addEventListener("click", () => {{
    document.querySelectorAll("[data-n]").forEach(cb => cb.checked = false);
    updateYBounds();
    render();
  }});

  document.getElementById("btnSelectAllIters").addEventListener("click", () => {{
    document.querySelectorAll("[data-iter]").forEach(cb => cb.checked = true);
    updateYBounds();
    render();
  }});
  document.getElementById("btnUnselectAllIters").addEventListener("click", () => {{
    document.querySelectorAll("[data-iter]").forEach(cb => cb.checked = false);
    updateYBounds();
    render();
  }});

  document.getElementById("btnExport").addEventListener("click", exportSVG);
}}

function zoomAt(factor, mouseY) {{
  const clampedY = Math.max(MARGIN.top, Math.min(MARGIN.top + PLOT_HEIGHT, mouseY));
  const ratioY = 1.0 - (clampedY - MARGIN.top) / PLOT_HEIGHT;
  const ySpan = (currentYMax - currentYMin) / factor;
  const yCenter = currentYMin + ratioY * (currentYMax - currentYMin);

  currentYMin = yCenter - ratioY * ySpan;
  currentYMax = yCenter + (1.0 - ratioY) * ySpan;

  render();
}}

function resetView() {{
  currentYMin = baseAutoYMin;
  currentYMax = baseAutoYMax;
  render();
}}

function exportSVG() {{
  if (currentActiveTab === "benchmark") {{
    hideCursorGuideline();
    const svgEl = document.getElementById("main-chart");
    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(svgEl);

    if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {{
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
  }} else {{
    exportSubSVG("test");
  }}
}}

function exportSubSVG(subType) {{
  hideTrainCursorGuideline();
  hideTestCursorGuideline();
  hideSubclassCursorGuideline();
  let svgId = "test-chart";
  if (subType === "train") svgId = "train-chart";
  else if (subType === "subclass") svgId = "subclass-chart";
  const svgEl = document.getElementById(svgId);
  if (!svgEl) return;
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svgEl);

  if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {{
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
</script>

</body>
</html>
"""
    return html_template


def find_dataset_root(outs_dir: Path) -> Optional[Path]:
    """Dynamically finds the real dataset directory on disk without hardcoding."""
    outs_dir = Path(outs_dir)
    # 1. Check split_summary if it has source_root
    sum_f = outs_dir / "data_splits" / "split_summary.json"
    if sum_f.is_file():
        try:
            data = json.loads(sum_f.read_text(encoding="utf-8"))
            s_root = data.get("source_root")
            if s_root and Path(s_root).is_dir():
                return Path(s_root)
        except Exception:
            pass

    # 2. Check test_full.txt common parent
    test_f = outs_dir / "data_splits" / "test_full.txt"
    if test_f.is_file():
        try:
            lines = [line.strip().split("\t")[0] for line in test_f.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                common = os.path.commonpath(lines)
                common_p = Path(common)
                if common_p.is_dir():
                    return common_p
        except Exception:
            pass

    # 3. Fallback to /data/wt/data/{dataset_name}
    cand = Path(f"/data/wt/data/{outs_dir.name}")
    if cand.is_dir():
        return cand
    return None


def scan_dataset_subclasses(root: Optional[Path]) -> List[Dict[str, Any]]:
    """Scans the real filesystem directory to get subclasses distribution."""
    if not root or not root.is_dir():
        return []
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    subclasses = []
    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    is_split_by_ok_ng = set(p.name.lower() for p in subdirs).issubset({"ok", "ng", "good", "bad", "test", "train"})

    if not is_split_by_ok_ng and subdirs:
        for sub in subdirs:
            img_dir = sub / "images" if (sub / "images").is_dir() else sub
            ann_dir = sub / "annotations" if (sub / "annotations").is_dir() else sub
            imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            normal_cnt = 0
            defect_cnt = 0
            for img in imgs:
                json_f = ann_dir / (img.stem + ".json")
                if json_f.is_file():
                    try:
                        data = json.loads(json_f.read_text(encoding="utf-8"))
                        shapes = data.get("shapes", [])
                        if len(shapes) > 0:
                            defect_cnt += 1
                        else:
                            normal_cnt += 1
                    except Exception:
                        normal_cnt += 1
                else:
                    normal_cnt += 1
            subclasses.append({
                "name": sub.name,
                "normal": normal_cnt,
                "defect": defect_cnt,
                "total": normal_cnt + defect_cnt
            })
    else:
        normal_cnt = 0
        defect_cnt = 0
        for sub in subdirs:
            imgs = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            if sub.name.lower() in {"ok", "good"}:
                normal_cnt += len(imgs)
            elif sub.name.lower() in {"ng", "bad"}:
                defect_cnt += len(imgs)
        subclasses.append({
            "name": root.name,
            "normal": normal_cnt,
            "defect": defect_cnt,
            "total": normal_cnt + defect_cnt
        })
    return subclasses


def scan_train_subclasses(outs_dir: Path) -> Dict[str, Any]:
    """Dynamically parses train sample distribution per subclass for each N size from disk."""
    outs_dir = Path(outs_dir)
    p = outs_dir / "data_splits"
    if not p.is_dir():
        return {}

    # 1. Check split_summary.json
    sum_f = p / "split_summary.json"
    if sum_f.is_file():
        try:
            data = json.loads(sum_f.read_text(encoding="utf-8"))
            subdirs = data.get("subdirectories", {})
            ns = data.get("train_sizes", [])
            if subdirs and ns:
                sub_list = []
                for sname in sorted(subdirs.keys()):
                    quotas = {str(k): int(v) for k, v in subdirs[sname].get("quotas", {}).items()}
                    sub_list.append({"name": sname, "counts": quotas})
                return {"ns": ns, "subclasses": sub_list}
        except Exception:
            pass

    # 2. Parse train_n*.txt directly
    train_files = sorted(p.glob("train_n*.txt"))
    if not train_files:
        return {}

    file_map = {}
    ns = []
    for tf in train_files:
        m = re.search(r"train_n(\d+)\.txt", tf.name)
        if m:
            n_val = int(m.group(1))
            ns.append(n_val)
            file_map[n_val] = tf
    ns = sorted(ns)

    sub_counts_by_n = {}
    all_subs = set()
    for n_val in ns:
        tf = file_map[n_val]
        counts = {}
        for line in tf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = Path(line).parts
            if len(parts) >= 2:
                sub_name = parts[-3] if parts[-2] == "images" else parts[-2]
            else:
                sub_name = "train"
            counts[sub_name] = counts.get(sub_name, 0) + 1
            all_subs.add(sub_name)
        sub_counts_by_n[str(n_val)] = counts

    sub_list = []
    for sname in sorted(all_subs):
        q = {str(n): sub_counts_by_n[str(n)].get(sname, 0) for n in ns}
        sub_list.append({"name": sname, "counts": q})

    return {"ns": ns, "subclasses": sub_list}


def scan_test_subclasses(outs_dir: Path, real_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Dynamically parses test sample distribution per subclass from test_full.txt."""
    outs_dir = Path(outs_dir)
    tf = outs_dir / "data_splits" / "test_full.txt"
    if not tf.is_file():
        return []
    try:
        lines = [line.strip().split("\t") for line in tf.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return []

        base_dir = None
        if real_root and real_root.is_dir():
            base_dir = str(real_root)
        else:
            paths = [Path(l[0]) for l in lines]
            base_dir = os.path.commonpath([str(p) for p in paths])

        subclass_stats = {}
        for l in lines:
            p_str = l[0]
            label = int(l[1]) if len(l) > 1 else 0
            if base_dir and p_str.startswith(base_dir):
                rel = os.path.relpath(p_str, base_dir)
            else:
                rel = p_str
            parts = rel.split(os.sep)
            if not parts or parts[0].lower() in ["ok", "ng", "good", "bad", "images", "annotations"]:
                sname = outs_dir.name
            else:
                sname = parts[0]
            if sname not in subclass_stats:
                subclass_stats[sname] = {"name": sname, "normal": 0, "defect": 0, "total": 0}
            if label == 0:
                subclass_stats[sname]["normal"] += 1
            else:
                subclass_stats[sname]["defect"] += 1
            subclass_stats[sname]["total"] += 1

        return [subclass_stats[k] for k in sorted(subclass_stats.keys())]
    except Exception as e:
        print(f"[warn] Failed to parse test subclasses: {e}")
        return []


def get_dataset_split_stats(outs_dir: Path) -> Dict[str, Any]:
    """Extracts train sample sizes, test normal/defect counts, and scans subclasses strictly from real filesystem directory."""
    sum_f = outs_dir / "final_multisize_summary.json"
    stats = {
        "name": outs_dir.name,
        "dataset_total": 0,
        "train_ns": [],
        "train_counts": {},
        "test_defect": 0,
        "test_normal": 0,
        "test_total": 0,
        "subclasses": [],
        "train_subclasses": {},
        "test_subclasses": []
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

                # Determine overall dataset total images
                dataset_total = None
                split_f = outs_dir / "data_splits" / "split_summary.json"
                if split_f.is_file():
                    try:
                        sdata = json.loads(split_f.read_text(encoding="utf-8"))
                        dataset_total = sdata.get("total_images")
                    except Exception:
                        pass

                if dataset_total is None:
                    test_f = outs_dir / "data_splits" / "test_full.txt"
                    train_fs = sorted((outs_dir / "data_splits").glob("train_n*.txt"))
                    if test_f.is_file() and train_fs:
                        try:
                            test_cnt = len(test_f.read_text(encoding="utf-8").splitlines())
                            max_train_cnt = max(len(tf.read_text(encoding="utf-8").splitlines()) for tf in train_fs)
                            dataset_total = test_cnt + max_train_cnt
                        except Exception:
                            pass

                if dataset_total is None and total_test > 0:
                    max_n = max(ns) if ns else 0
                    dataset_total = total_test + max_n

                stats["dataset_total"] = dataset_total or 0

                # Dynamically scan real directory for subclasses
                real_root = find_dataset_root(outs_dir)
                stats["subclasses"] = scan_dataset_subclasses(real_root)
                stats["train_subclasses"] = scan_train_subclasses(outs_dir)
                stats["test_subclasses"] = scan_test_subclasses(outs_dir, real_root)
                if (stats["dataset_total"] == 0 or stats["dataset_total"] is None) and stats["subclasses"]:
                    stats["dataset_total"] = sum(s["total"] for s in stats["subclasses"])
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
    
    # Enrich dataset_data with real_vram_measurements.json if available
    vram_file = outs_dir / "real_vram_measurements.json"
    if vram_file.is_file():
        try:
            vram_data = json.loads(vram_file.read_text(encoding="utf-8"))
            train_vram = vram_data.get("train", {})
            infer_vram = vram_data.get("infer", {})
            for r in dataset_data:
                sz = str(r.get("size"))
                n = str(r.get("n"))
                if infer_vram and sz in infer_vram:
                    if n in infer_vram[sz].get("dino", {}):
                        r["din_vram_gb"] = infer_vram[sz]["dino"][n]
                    if n in infer_vram[sz].get("patch", {}):
                        r["pat_vram_gb"] = infer_vram[sz]["patch"][n]
                    if n in infer_vram[sz].get("e2e", {}):
                        r["e2e_vram_gb"] = infer_vram[sz]["e2e"][n]
                if not r.get("din_train_vram_gb"):
                    r["din_train_vram_gb"] = train_vram.get(sz, {}).get("dino", {}).get(n, 0)
                if not r.get("pat_train_vram_gb"):
                    r["pat_train_vram_gb"] = train_vram.get(sz, {}).get("patch", {}).get(n, 0)
                if not r.get("e2e_train_vram_gb"):
                    r["e2e_train_vram_gb"] = train_vram.get(sz, {}).get("e2e", {}).get(n, r.get("din_train_vram_gb", 0))
        except Exception:
            pass

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
