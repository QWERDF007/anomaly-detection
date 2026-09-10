#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

def build_interactive_html(
    dataset_name: str,
    dataset_data: List[Dict[str, Any]],
    has_bank: bool = False,
    all_datasets_stats: Optional[Dict[str, Any]] = None
) -> str:
    """Generates strictly ONE self-contained interactive benchmark dashboard HTML for a single dataset."""
    json_data = json.dumps(dataset_data, ensure_ascii=False)
    has_bank_js = "true" if has_bank else "false"
    all_datasets_json = json.dumps(all_datasets_stats or {}, ensure_ascii=False)

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
  }}
  .header-tabs .tab-btn:hover {{
    color: #0f172a;
    background: rgba(255, 255, 255, 0.7);
  }}
  .header-tabs .tab-btn.active {{
    background: #ffffff;
    color: #2563eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  }}
  .tab-badge {{
    background: #ef4444;
    color: white;
    font-size: 0.65rem;
    padding: 1px 6px;
    border-radius: 10px;
    font-weight: 700;
  }}
  .preset-btn {{
    width: 100%;
    text-align: center;
    justify-content: center;
    font-size: 0.76rem;
    padding: 5px 8px;
    border-radius: 5px;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #334155;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .preset-btn:hover {{
    background: #e2e8f0;
    color: #0f172a;
  }}
  .preset-btn-accent {{
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1d4ed8;
    font-weight: 600;
  }}
  .preset-btn-accent:hover {{
    background: #dbeafe;
    color: #1e40af;
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

  /* Chart Area */
  .chart-container {{
    flex: 1;
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

  .chart-body {{
    flex: 1;
    min-height: 0;
    position: relative;
    width: 100%;
    height: 100%;
    cursor: grab;
    user-select: none;
    background: #ffffff;
    overflow: hidden;
  }}
  .chart-body:active {{
    cursor: grabbing;
  }}

  svg#main-chart {{
    width: 100%;
    height: 100%;
    display: block;
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
      <button class="tab-btn active" id="tabBtnBenchmark" title="查看模型指标评测">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 9l-5 5-4-4-6 6"/></svg>
        <span>算法模型基准评测</span>
      </button>
      <button class="tab-btn" id="tabBtnDataset" title="查看训练集与测试集数量规模分布">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
        <span>训练集与测试集数量分布</span>
        <span class="tab-badge">图表</span>
      </button>
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
      <label for="dsViewMode">展示模式:</label>
      <select id="dsViewMode" class="custom-select" style="min-width: 260px;">
        <option value="detailed" selected>全量明细 (训练良品各N / 测试良品 / 测试缺陷 / 测试总量)</option>
        <option value="totals">总量对比 (训练集最大样本量 vs 测试集总样本量)</option>
        <option value="stacked">测试集构成比例 (良品 vs 缺陷 100% 堆叠)</option>
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
    <!-- 1. Controls for Benchmark Tab -->
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

    <!-- 2. Controls for Dataset Splits Tab -->
    <div id="datasetControls" style="display: none; flex-direction: column; gap: 8px;">
      <div class="control-group">
        <h3>
          <span>对比数据集 (Datasets)</span>
          <div class="quick-links-group">
            <span class="quick-link" id="btnSelectAllDs">全选</span>
            <span class="quick-link-sep">/</span>
            <span class="quick-link" id="btnUnselectAllDs">全不选</span>
          </div>
        </h3>
        <div class="checkbox-list" id="dsListContainer">
        </div>
        <div style="margin-top: 8px; display: flex; flex-direction: column; gap: 5px;" id="dsQuickButtonsContainer">
        </div>
      </div>

      <div class="divider"></div>

      <div class="control-group">
        <h3>
          <span>包含统计类别 (Series)</span>
        </h3>
        <div class="checkbox-list">
          <label class="checkbox-item">
            <input type="checkbox" id="chkDsTrain" checked>
            <span class="color-dot" style="background: #2563eb;"></span>
            <span>训练集良品样本量 (Train N)</span>
          </label>
          <label class="checkbox-item">
            <input type="checkbox" id="chkDsTestNormal" checked>
            <span class="color-dot" style="background: #10b981;"></span>
            <span>测试集良品数量 (Test OK)</span>
          </label>
          <label class="checkbox-item">
            <input type="checkbox" id="chkDsTestDefect" checked>
            <span class="color-dot" style="background: #f43f5e;"></span>
            <span>测试集缺陷数量 (Test NG)</span>
          </label>
          <label class="checkbox-item">
            <input type="checkbox" id="chkDsTestTotal" checked>
            <span class="color-dot" style="background: #64748b;"></span>
            <span>测试集总样本量 (Test Total)</span>
          </label>
        </div>
      </div>

      <div class="divider"></div>

      <div class="control-group">
        <h3><span>说明与规范</span></h3>
        <div style="font-size: 0.77rem; color: #64748b; line-height: 1.45;">
          • <strong>训练集</strong>: 单分类异常检测仅采集纯良品图像。<br>
          • <strong>测试集</strong>: 包含未知良品与真实缺陷图像。<br>
          • <strong>同屏对比</strong>: 支持多个数据集在此图表中横向并列对比。
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
      <svg id="main-chart" viewBox="0 0 1300 660" preserveAspectRatio="xMidYMid meet">
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

      <!-- Tooltip -->
      
    </div>

    <div class="detail-panel" id="detailPanel">
      <span>💡 <strong>交互指引:</strong> 顶部下拉列表可自由切换各项评测指标与观察维度。鼠标移动处提供<strong>水平辅助虚线与实时数值标尺</strong>，柱状图上方直观标示精确数值，悬停查看完整指标卡片，支持滚轮缩放与拖拽平移。</span>
    </div>
  
    </div>

    <!-- 2. Dataset Splits Distribution Chart Area -->
    <div id="datasetChartArea" style="display: none; flex-direction: column; width: 100%; height: 100%; min-height: 0;">
      <div class="chart-header">
        <div class="title-area">
          <div class="chart-title" id="dsChartTitle">📦 数据集样本量分布看板 (训练集良品规模 vs 测试集良品/缺陷数量)</div>
          <div class="chart-stats" id="dsChartStats">展示选中数据集的样本规模对比</div>
        </div>
        <div class="chart-actions">
          <button class="btn" id="btnDsResetView" title="重置初始视图">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
            重置
          </button>
        </div>
      </div>

      <div class="chart-body" id="dsChartBody">
        <svg id="dataset-chart" viewBox="0 0 1300 660" preserveAspectRatio="xMidYMid meet">
          <defs>
            <clipPath id="ds-chart-clip">
              <rect id="dsClipRect" x="85" y="65" width="1165" height="530" />
            </clipPath>
          </defs>

          <!-- Grid & Axes -->
          <g id="ds-grid-group"></g>
          <g id="ds-axes-group"></g>

          <!-- Data Bars -->
          <g id="ds-data-group" clip-path="url(#ds-chart-clip)">
            <g id="ds-bars-group"></g>
          </g>

          <!-- Legends -->
          <g id="ds-legend-group"></g>

          <!-- Mouse Guideline -->
          <g id="ds-cursor-group" pointer-events="none" style="display: none;">
            <line id="dsCursorHLine" x1="0" y1="0" x2="0" y2="0" stroke="#334155" stroke-width="1.3" stroke-dasharray="5 4" style="filter: drop-shadow(0 0 1.5px rgba(255,255,255,0.95));" />
            <g id="dsCursorBadge">
              <rect id="dsCursorBadgeRect" x="0" y="0" width="60" height="20" rx="3" fill="#0f172a" stroke="#3b82f6" stroke-width="1" />
              <text id="dsCursorBadgeText" x="0" y="0" fill="#f8fafc" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-weight="600" text-anchor="middle" dominant-baseline="central"></text>
            </g>
          </g>
        </svg>
      </div>

      <div class="detail-panel" id="dsDetailPanel">
        <span>💡 <strong>样本量统计指引:</strong> 左侧可自由勾选需要对比的数据集（默认已将同批次数据集全部加入对比），点击「仅看两个台达散热片数据集」可在同一图表中横向并列对比。鼠标悬停柱子查看详细数量与占比，光标提供水平辅助虚线与实时数量标尺。</span>
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
const ALL_DATASETS_STATS = {all_datasets_json};
let currentActiveTab = "benchmark";
let currentDsYMax = 1000;

// Color scheme
const COLOR_E2E = "#059669";
const COLOR_DINO = "#2563eb";
const COLOR_PATCH = "#ea580c";

// Chart Layout Boundaries
const SVG_WIDTH = 1300;
const SVG_HEIGHT = 660;
const MARGIN = {{ top: 65, right: 50, bottom: 65, left: 85 }};
const PLOT_WIDTH = SVG_WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

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
  fp: {{
    label: "False Positives (测试集良品误报数 FP)",
    title: "测试集未见良品误报数量 (False Positives / FP, 越低越好)",
    unit: " 张",
    digits: 0,
    yLabel: "测试集良品误报数 (FP / 张)",
    field: "fp",
    higherIsBetter: false,
    defaultYRange: null,
    isRate: false
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
  vram_gb: {{
    label: "GPU VRAM (推理显存占用 GB)",
    title: "GPU 推理显存峰值 (Inference Peak VRAM, 越低越好)",
    unit: " GB",
    digits: 2,
    yLabel: "显存占用 (GB)",
    field: "vram_gb",
    higherIsBetter: false,
    defaultYRange: null,
    isRate: false
  }}
}};

const NON_ITERATION_METRICS = ["vram_gb", "lat_ms", "fps"];

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
  setupEventListeners();
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
    <optgroup label="① 测试集评测指标 (泛化表现 / 缺陷检出率与漏报误报率)">
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
    </optgroup>
    <optgroup label="③ 硬件与推理效能 (按样本量 / 尺寸 / 模型观察)">
      <option value="lat_ms">Inference Latency (单图端到端推理时延 ms - 越低越好)</option>
      <option value="fps">Inference FPS (实际推理吞吐量 FPS - 越高越好)</option>
      <option value="vram_gb">GPU VRAM (推理显存占用峰值 GB - 越低越好)</option>
    </optgroup>
  `;
  sel.value = currentMetric;
}}

function initTabs() {{
  const btnB = document.getElementById("tabBtnBenchmark");
  const btnD = document.getElementById("tabBtnDataset");
  const bSel = document.getElementById("benchmarkSelectors");
  const dSel = document.getElementById("datasetSelectors");
  const bCtl = document.getElementById("benchmarkControls");
  const dCtl = document.getElementById("datasetControls");
  const bArea = document.getElementById("benchmarkChartArea");
  const dArea = document.getElementById("datasetChartArea");

  btnB.addEventListener("click", () => {{
    currentActiveTab = "benchmark";
    btnB.classList.add("active");
    btnD.classList.remove("active");
    bSel.style.display = "flex";
    dSel.style.display = "none";
    bCtl.style.display = "flex";
    dCtl.style.display = "none";
    bArea.style.display = "flex";
    dArea.style.display = "none";
    hideTooltip();
    hideCursorGuideline();
    hideDsCursorGuideline();
    render();
  }});

  btnD.addEventListener("click", () => {{
    currentActiveTab = "dataset";
    btnD.classList.add("active");
    btnB.classList.remove("active");
    dSel.style.display = "flex";
    bSel.style.display = "none";
    dCtl.style.display = "flex";
    bCtl.style.display = "none";
    dArea.style.display = "flex";
    bArea.style.display = "none";
    hideTooltip();
    hideCursorGuideline();
    hideDsCursorGuideline();
    renderDatasetChart();
  }});
}}

function initDatasetControls() {{
  const container = document.getElementById("dsListContainer");
  const quickContainer = document.getElementById("dsQuickButtonsContainer");
  if (!container || !ALL_DATASETS_STATS) return;

  container.innerHTML = "";
  const dsNames = Object.keys(ALL_DATASETS_STATS);

  dsNames.forEach(name => {{
    const ds = ALL_DATASETS_STATS[name];
    const label = document.createElement("label");
    label.className = "checkbox-item";
    label.innerHTML = `
      <input type="checkbox" checked data-ds-name="${{name}}">
      <span class="color-dot" style="background: ${{ds.is_current ? '#2563eb' : '#64748b'}};"></span>
      <span>${{name}} ${{ds.is_current ? '<strong>(当前)</strong>' : ''}}</span>
    `;
    label.querySelector("input").addEventListener("change", renderDatasetChart);
    container.appendChild(label);
  }});

  quickContainer.innerHTML = "";

  const hasDeltaBoth = dsNames.includes("台达散热片栅格") && dsNames.includes("台达散热片正反面");
  if (hasDeltaBoth) {{
    const btnDelta = document.createElement("button");
    btnDelta.className = "preset-btn preset-btn-accent";
    btnDelta.innerHTML = "⚡ 仅看两个台达散热片数据集 (栅格 + 正反面)";
    btnDelta.addEventListener("click", () => {{
      document.querySelectorAll("[data-ds-name]").forEach(cb => {{
        const n = cb.getAttribute("data-ds-name");
        cb.checked = (n === "台达散热片栅格" || n === "台达散热片正反面");
      }});
      renderDatasetChart();
    }});
    quickContainer.appendChild(btnDelta);
  }}

  const btnOnlyCur = document.createElement("button");
  btnOnlyCur.className = "preset-btn";
  btnOnlyCur.innerHTML = `🎯 仅看当前数据集 (${{DATASET_NAME}})`;
  btnOnlyCur.addEventListener("click", () => {{
    document.querySelectorAll("[data-ds-name]").forEach(cb => {{
      cb.checked = (cb.getAttribute("data-ds-name") === DATASET_NAME);
    }});
    renderDatasetChart();
  }});
  quickContainer.appendChild(btnOnlyCur);

  document.getElementById("btnSelectAllDs").addEventListener("click", () => {{
    document.querySelectorAll("[data-ds-name]").forEach(cb => cb.checked = true);
    renderDatasetChart();
  }});
  document.getElementById("btnUnselectAllDs").addEventListener("click", () => {{
    document.querySelectorAll("[data-ds-name]").forEach(cb => cb.checked = false);
    renderDatasetChart();
  }});

  ["chkDsTrain", "chkDsTestNormal", "chkDsTestDefect", "chkDsTestTotal"].forEach(id => {{
    document.getElementById(id).addEventListener("change", renderDatasetChart);
  }});
  document.getElementById("dsViewMode").addEventListener("change", renderDatasetChart);

  const dsChartBody = document.getElementById("dsChartBody");
  dsChartBody.addEventListener("mousemove", (e) => updateDsCursorGuideline(e));
  dsChartBody.addEventListener("mouseleave", () => hideDsCursorGuideline());
  document.getElementById("btnDsResetView").addEventListener("click", renderDatasetChart);
}}

function renderDatasetChart() {{
  const barsGroup = document.getElementById("ds-bars-group");
  const gridGroup = document.getElementById("ds-grid-group");
  const axesGroup = document.getElementById("ds-axes-group");
  const legendGroup = document.getElementById("ds-legend-group");
  if (!barsGroup || !gridGroup || !axesGroup || !legendGroup) return;

  barsGroup.innerHTML = "";
  gridGroup.innerHTML = "";
  axesGroup.innerHTML = "";
  legendGroup.innerHTML = "";

  const selectedDsNames = [];
  document.querySelectorAll("[data-ds-name]").forEach(cb => {{
    if (cb.checked) selectedDsNames.push(cb.getAttribute("data-ds-name"));
  }});

  const titleEl = document.getElementById("dsChartTitle");
  const statsEl = document.getElementById("dsChartStats");

  if (selectedDsNames.length === 0) {{
    if (titleEl) titleEl.textContent = "请在左侧勾选至少一个数据集";
    if (statsEl) statsEl.textContent = "当前未选择数据集";
    return;
  }}

  const viewMode = document.getElementById("dsViewMode").value;
  const incTrain = document.getElementById("chkDsTrain").checked;
  const incNormal = document.getElementById("chkDsTestNormal").checked;
  const incDefect = document.getElementById("chkDsTestDefect").checked;
  const incTotal = document.getElementById("chkDsTestTotal").checked;

  const datasetItems = [];
  let globalMaxCount = 0;

  selectedDsNames.forEach(name => {{
    const ds = ALL_DATASETS_STATS[name];
    if (!ds) return;

    const series = [];
    if (viewMode === "detailed") {{
      if (incTrain && ds.train_ns) {{
        ds.train_ns.forEach(n => {{
          const c = n === 100 ? "#93c5fd" : (n === 200 ? "#3b82f6" : "#1d4ed8");
          series.push({{
            type: "train",
            category: `训练集良品 (N=${{n}})`,
            shortLabel: `训练 N=${{n}}`,
            val: n,
            color: c,
            dsName: name
          }});
        }});
      }}
      if (incNormal) {{
        series.push({{
          type: "test_normal",
          category: "测试集良品 (OK)",
          shortLabel: "测试良品",
          val: ds.test_normal,
          color: "#10b981",
          dsName: name
        }});
      }}
      if (incDefect) {{
        series.push({{
          type: "test_defect",
          category: "测试集缺陷 (NG)",
          shortLabel: "测试缺陷",
          val: ds.test_defect,
          color: "#f43f5e",
          dsName: name
        }});
      }}
      if (incTotal) {{
        series.push({{
          type: "test_total",
          category: "测试集总样本量 (Total)",
          shortLabel: "测试总量",
          val: ds.test_total,
          color: "#64748b",
          dsName: name
        }});
      }}
    }} else if (viewMode === "totals") {{
      const maxN = ds.train_ns && ds.train_ns.length > 0 ? Math.max(...ds.train_ns) : 400;
      series.push({{
        type: "train_max",
        category: `训练集最大样本量 (N=${{maxN}})`,
        shortLabel: `训练 (N=${{maxN}})`,
        val: maxN,
        color: "#2563eb",
        dsName: name
      }});
      series.push({{
        type: "test_total",
        category: "测试集总样本量",
        shortLabel: "测试总量",
        val: ds.test_total,
        color: "#7c3aed",
        dsName: name
      }});
    }} else if (viewMode === "stacked") {{
      series.push({{
        type: "stacked",
        category: "测试集良品/缺陷构成",
        shortLabel: "测试集构成",
        normalVal: ds.test_normal,
        defectVal: ds.test_defect,
        totalVal: ds.test_total,
        normalPct: (ds.test_normal / ds.test_total) * 100,
        defectPct: (ds.test_defect / ds.test_total) * 100,
        dsName: name
      }});
    }}

    series.forEach(s => {{
      if (s.val !== undefined && s.val > globalMaxCount) globalMaxCount = s.val;
      if (s.totalVal !== undefined && s.totalVal > globalMaxCount) globalMaxCount = s.totalVal;
    }});

    datasetItems.push({{
      name: name,
      meta: ds,
      series: series
    }});
  }});

  const isStacked = (viewMode === "stacked");
  const yMax = isStacked ? 100 : (globalMaxCount > 0 ? Math.ceil(globalMaxCount * 1.18) : 1000);
  currentDsYMax = yMax;
  const yMin = 0;
  const valSpan = yMax - yMin;

  function dsValToY(val) {{
    if (valSpan <= 0) return MARGIN.top + PLOT_HEIGHT;
    const ratio = Math.max(0, Math.min(1, (val - yMin) / valSpan));
    return MARGIN.top + (1.0 - ratio) * PLOT_HEIGHT;
  }}

  if (titleEl) {{
    titleEl.textContent = isStacked 
      ? "📦 数据集样本量分布看板 (测试集良品 vs 缺陷 100% 堆叠占比)" 
      : "📦 数据集样本量分布看板 (训练集良品规模 vs 测试集样本量)";
  }}
  if (statsEl) {{
    statsEl.textContent = `展示 ${{selectedDsNames.length}} 个数据集样本规模对比 | Y 轴区间: 0 ~ ${{yMax.toLocaleString()}} ${{isStacked ? "%" : "张"}}`;
  }}

  // 1. Grid & Y Axis
  const tickCount = 6;
  const step = yMax / tickCount;
  for (let i = 0; i <= tickCount; i++) {{
    const yVal = Math.round(i * step);
    const py = dsValToY(yVal);
    if (py < MARGIN.top - 2 || py > MARGIN.top + PLOT_HEIGHT + 2) continue;

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
    text.textContent = yVal.toLocaleString() + (isStacked ? "%" : "");
    axesGroup.appendChild(text);
  }}

  const ySpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ySpine.setAttribute("x1", MARGIN.left);
  ySpine.setAttribute("y1", MARGIN.top);
  ySpine.setAttribute("x2", MARGIN.left);
  ySpine.setAttribute("y2", MARGIN.top + PLOT_HEIGHT);
  ySpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(ySpine);

  const yTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yTitle.setAttribute("transform", "rotate(-90)");
  yTitle.setAttribute("x", -(MARGIN.top + PLOT_HEIGHT / 2));
  yTitle.setAttribute("y", MARGIN.left - 54);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-title");
  yTitle.textContent = isStacked ? "测试集样本构成占比 (%)" : "样本图像数量 (张)";
  axesGroup.appendChild(yTitle);

  const xSpine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xSpine.setAttribute("x1", MARGIN.left);
  xSpine.setAttribute("y1", MARGIN.top + PLOT_HEIGHT);
  xSpine.setAttribute("x2", MARGIN.left + PLOT_WIDTH);
  xSpine.setAttribute("y2", MARGIN.top + PLOT_HEIGHT);
  xSpine.setAttribute("class", "axis-line");
  axesGroup.appendChild(xSpine);

  // 2. Draw Dataset Slots along X Axis
  const numSlots = datasetItems.length;
  const slotW = PLOT_WIDTH / numSlots;

  datasetItems.forEach((dsItem, slotIdx) => {{
    const slotLeft = MARGIN.left + slotIdx * slotW;
    const slotCenterX = slotLeft + slotW / 2;

    if (slotIdx > 0) {{
      const divLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      divLine.setAttribute("x1", slotLeft);
      divLine.setAttribute("y1", MARGIN.top);
      divLine.setAttribute("x2", slotLeft);
      divLine.setAttribute("y2", MARGIN.top + PLOT_HEIGHT);
      divLine.setAttribute("stroke", "#cbd5e1");
      divLine.setAttribute("stroke-width", "1.2");
      divLine.setAttribute("stroke-dasharray", "4 3");
      axesGroup.appendChild(divLine);
    }}

    const dsNameText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    dsNameText.setAttribute("x", slotCenterX);
    dsNameText.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 22);
    dsNameText.setAttribute("text-anchor", "middle");
    dsNameText.setAttribute("font-size", "13");
    dsNameText.setAttribute("font-weight", "700");
    dsNameText.setAttribute("fill", dsItem.meta.is_current ? "#1d4ed8" : "#0f172a");
    dsNameText.textContent = dsItem.name + (dsItem.meta.is_current ? " (当前看板)" : "");
    axesGroup.appendChild(dsNameText);

    const dsSubText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    dsSubText.setAttribute("x", slotCenterX);
    dsSubText.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 39);
    dsSubText.setAttribute("text-anchor", "middle");
    dsSubText.setAttribute("font-size", "10.5");
    dsSubText.setAttribute("fill", "#64748b");
    const okPct = (dsItem.meta.test_normal / dsItem.meta.test_total * 100).toFixed(1);
    const ngPct = (dsItem.meta.test_defect / dsItem.meta.test_total * 100).toFixed(1);
    dsSubText.textContent = `测试集总量 ${{dsItem.meta.test_total.toLocaleString()}} 张 (${{okPct}}% 良品 / ${{ngPct}}% 缺陷)`;
    axesGroup.appendChild(dsSubText);

    const seriesList = dsItem.series;
    if (seriesList.length === 0) return;

    if (isStacked) {{
      const barW = Math.min(130, slotW * 0.46);
      const bx = slotCenterX - barW / 2;
      const s = seriesList[0];
      const normH = (s.normalPct / 100) * PLOT_HEIGHT;
      const defH = (s.defectPct / 100) * PLOT_HEIGHT;
      const normY = MARGIN.top + PLOT_HEIGHT - normH;
      const defY = normY - defH;

      const rNorm = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rNorm.setAttribute("x", bx);
      rNorm.setAttribute("y", normY);
      rNorm.setAttribute("width", barW);
      rNorm.setAttribute("height", normH);
      rNorm.setAttribute("fill", "#10b981");
      rNorm.setAttribute("rx", "2");
      rNorm.setAttribute("class", "bar-base");
      rNorm.addEventListener("mouseenter", (e) => showDsTooltip(e, dsItem, "测试集良品", s.normalVal, s.normalPct));
      rNorm.addEventListener("mousemove", (e) => {{ updateTooltipPos(e); updateDsCursorGuideline(e); }});
      rNorm.addEventListener("mouseleave", hideTooltip);
      barsGroup.appendChild(rNorm);

      const rDef = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rDef.setAttribute("x", bx);
      rDef.setAttribute("y", defY);
      rDef.setAttribute("width", barW);
      rDef.setAttribute("height", defH);
      rDef.setAttribute("fill", "#f43f5e");
      rDef.setAttribute("rx", "2");
      rDef.setAttribute("class", "bar-base");
      rDef.addEventListener("mouseenter", (e) => showDsTooltip(e, dsItem, "测试集缺陷", s.defectVal, s.defectPct));
      rDef.addEventListener("mousemove", (e) => {{ updateTooltipPos(e); updateDsCursorGuideline(e); }});
      rDef.addEventListener("mouseleave", hideTooltip);
      barsGroup.appendChild(rDef);

      if (normH > 22) {{
        const tNorm = document.createElementNS("http://www.w3.org/2000/svg", "text");
        tNorm.setAttribute("x", slotCenterX);
        tNorm.setAttribute("y", normY + normH / 2 + 4);
        tNorm.setAttribute("text-anchor", "middle");
        tNorm.setAttribute("fill", "#ffffff");
        tNorm.setAttribute("font-size", "11");
        tNorm.setAttribute("font-weight", "700");
        tNorm.textContent = `${{s.normalVal.toLocaleString()}} 张 (${{s.normalPct.toFixed(1)}}%)`;
        barsGroup.appendChild(tNorm);
      }}
      if (defH > 22) {{
        const tDef = document.createElementNS("http://www.w3.org/2000/svg", "text");
        tDef.setAttribute("x", slotCenterX);
        tDef.setAttribute("y", defY + defH / 2 + 4);
        tDef.setAttribute("text-anchor", "middle");
        tDef.setAttribute("fill", "#ffffff");
        tDef.setAttribute("font-size", "11");
        tDef.setAttribute("font-weight", "700");
        tDef.textContent = `${{s.defectVal.toLocaleString()}} 张 (${{s.defectPct.toFixed(1)}}%)`;
        barsGroup.appendChild(tDef);
      }}
    }} else {{
      const numBars = seriesList.length;
      const usableW = slotW * 0.82;
      const barW = Math.max(16, Math.min(65, (usableW / numBars) - 6));
      const totalBarsW = numBars * barW + (numBars - 1) * 6;
      const startX = slotCenterX - totalBarsW / 2;

      seriesList.forEach((s, barIdx) => {{
        const bx = startX + barIdx * (barW + 6);
        const bHeight = Math.max(2, (s.val / yMax) * PLOT_HEIGHT);
        const by = MARGIN.top + PLOT_HEIGHT - bHeight;

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", bx);
        rect.setAttribute("y", by);
        rect.setAttribute("width", barW);
        rect.setAttribute("height", bHeight);
        rect.setAttribute("fill", s.color);
        rect.setAttribute("rx", "3");
        rect.setAttribute("ry", "3");
        rect.setAttribute("class", "bar-base");

        rect.addEventListener("mouseenter", (e) => showDsTooltip(e, dsItem, s.category, s.val, null));
        rect.addEventListener("mousemove", (e) => {{ updateTooltipPos(e); updateDsCursorGuideline(e); }});
        rect.addEventListener("mouseleave", hideTooltip);
        rect.addEventListener("click", () => showDsDetailCard(dsItem, s));

        barsGroup.appendChild(rect);

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", bx + barW / 2);
        text.setAttribute("y", Math.max(MARGIN.top + 12, by - 6));
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", barW < 26 ? "9.5" : "11");
        text.setAttribute("font-weight", "700");
        text.setAttribute("fill", s.color);
        text.textContent = s.val.toLocaleString();
        barsGroup.appendChild(text);

        if (barW >= 22) {{
          const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
          sub.setAttribute("x", bx + barW / 2);
          sub.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 10);
          sub.setAttribute("text-anchor", "middle");
          sub.setAttribute("font-size", "8.5");
          sub.setAttribute("fill", "#64748b");
          sub.textContent = s.shortLabel.replace("训练 N=", "N=").replace("测试良品", "良品").replace("测试缺陷", "缺陷").replace("测试总量", "总数");
          barsGroup.appendChild(sub);
        }}
      }});
    }}
  }});

  renderDsLegends(legendGroup, viewMode, incTrain, incNormal, incDefect, incTotal);
  updateDsCursorGuideline();
}}

function renderDsLegends(group, viewMode, incTrain, incNormal, incDefect, incTotal) {{
  group.innerHTML = "";
  const legendItems = [];

  if (viewMode === "stacked") {{
    legendItems.push({{ label: "测试集良品 (OK)", color: "#10b981" }});
    legendItems.push({{ label: "测试集缺陷 (NG)", color: "#f43f5e" }});
  }} else if (viewMode === "totals") {{
    legendItems.push({{ label: "训练集最大样本量 (Train Max N)", color: "#2563eb" }});
    legendItems.push({{ label: "测试集总样本量 (Test Total)", color: "#7c3aed" }});
  }} else {{
    if (incTrain) {{
      legendItems.push({{ label: "训练良品 (N=100/200/400)", color: "#2563eb" }});
    }}
    if (incNormal) {{
      legendItems.push({{ label: "测试集良品 (OK)", color: "#10b981" }});
    }}
    if (incDefect) {{
      legendItems.push({{ label: "测试集缺陷 (NG)", color: "#f43f5e" }});
    }}
    if (incTotal) {{
      legendItems.push({{ label: "测试集总数 (Total)", color: "#64748b" }});
    }}
  }}

  let curX = MARGIN.left;
  legendItems.forEach(item => {{
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", curX);
    rect.setAttribute("y", MARGIN.top - 26);
    rect.setAttribute("width", 14);
    rect.setAttribute("height", 14);
    rect.setAttribute("rx", "3");
    rect.setAttribute("fill", item.color);
    g.appendChild(rect);

    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", curX + 18);
    txt.setAttribute("y", MARGIN.top - 15);
    txt.setAttribute("font-size", "11.5");
    txt.setAttribute("font-weight", "600");
    txt.setAttribute("fill", "#334155");
    txt.textContent = item.label;
    g.appendChild(txt);

    curX += item.label.length * 11 + 38;
    group.appendChild(g);
  }});
}}

function showDsTooltip(e, dsItem, categoryName, countVal, pctVal) {{
  const ds = dsItem.meta;
  const okPct = (ds.test_normal / ds.test_total * 100).toFixed(1);
  const ngPct = (ds.test_defect / ds.test_total * 100).toFixed(1);

  tooltip.innerHTML = `
    <div class="tt-title">
      <span style="display:inline-block; width: 10px; height: 10px; border-radius: 50%; background: #2563eb;"></span>
      <span>${{dsItem.name}} 样本规模画像</span>
    </div>
    <div class="tt-row tt-active-row">
      <span style="font-weight:700; color:#e0f2fe;">当前悬停项:</span>
      <span class="tt-val tt-highlight">${{categoryName}}: ${{countVal.toLocaleString()}} 张 ${{pctVal !== null ? `(${{pctVal.toFixed(1)}}%)` : ''}}</span>
    </div>
    <div class="divider" style="margin: 6px 0; opacity: 0.3;"></div>
    <div class="tt-row"><span style="color:#93c5fd; font-weight:700;">【训练集样本量 (Train Normal)】</span></div>
    <div class="tt-row"><span>训练正常样本梯度:</span> <span class="tt-val" style="color:#60a5fa; font-weight:700;">N = [${{ds.train_ns.join(", ")}}]</span></div>
    <div class="divider" style="margin: 6px 0; opacity: 0.3;"></div>
    <div class="tt-row"><span style="color:#93c5fd; font-weight:700;">【测试集样本量 (Test Full)】</span></div>
    <div class="tt-row"><span>测试集良品 (OK):</span> <span class="tt-val" style="color:#4ade80; font-weight:600;">${{ds.test_normal.toLocaleString()}} 张 (${{okPct}}%)</span></div>
    <div class="tt-row"><span>测试集缺陷 (NG):</span> <span class="tt-val" style="color:#f43f5e; font-weight:700;">${{ds.test_defect.toLocaleString()}} 张 (${{ngPct}}%)</span></div>
    <div class="tt-row"><span>测试集全量总数:</span> <span class="tt-val" style="color:#ffffff; font-weight:700;">${{ds.test_total.toLocaleString()}} 张</span></div>
  `;
  tooltip.style.display = "block";
  updateTooltipPos(e);
}}

function showDsDetailCard(dsItem, s) {{
  const ds = dsItem.meta;
  const panel = document.getElementById("dsDetailPanel");
  if (!panel) return;
  const okPct = (ds.test_normal / ds.test_total * 100).toFixed(1);
  const ngPct = (ds.test_defect / ds.test_total * 100).toFixed(1);
  panel.innerHTML = `
    <span><strong>选中数据集:</strong> <span style="color: #2563eb; font-weight: 700;">${{dsItem.name}}</span> | <strong>${{s.category}}:</strong> <span style="color: ${{s.color}}; font-weight: 700;">${{s.val.toLocaleString()}} 张</span> | <strong>训练集梯度:</strong> N=[${{ds.train_ns.join(", ")}}] | <strong>测试集全量:</strong> ${{ds.test_total.toLocaleString()}} 张 (良品: ${{ds.test_normal.toLocaleString()}} / ${{okPct}}% , 缺陷: ${{ds.test_defect.toLocaleString()}} / ${{ngPct}}%)</span>
  `;
}}

let lastDsMousePos = null;

function getDsSvgCoords(e) {{
  const svg = document.getElementById("dataset-chart");
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
    x: ((e.clientX - rect.left) / rect.width) * SVG_WIDTH,
    y: ((e.clientY - rect.top) / rect.height) * SVG_HEIGHT
  }};
}}

function updateDsCursorGuideline(e) {{
  if (currentActiveTab !== "dataset") return;
  if (e && e.clientX !== undefined) {{
    lastDsMousePos = {{ clientX: e.clientX, clientY: e.clientY }};
  }}
  if (!lastDsMousePos) {{
    hideDsCursorGuideline();
    return;
  }}

  const cursorGroup = document.getElementById("ds-cursor-group");
  const cursorHLine = document.getElementById("dsCursorHLine");
  const cursorBadge = document.getElementById("dsCursorBadge");
  const cursorBadgeRect = document.getElementById("dsCursorBadgeRect");
  const cursorBadgeText = document.getElementById("dsCursorBadgeText");
  if (!cursorGroup || !cursorHLine) return;

  const p = getDsSvgCoords(lastDsMousePos);
  if (!p) return;

  const inPlotY = p.y >= MARGIN.top && p.y <= MARGIN.top + PLOT_HEIGHT;
  const inPlotX = p.x >= MARGIN.left - 45 && p.x <= MARGIN.left + PLOT_WIDTH + 30;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";
    cursorHLine.setAttribute("x1", MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", MARGIN.left + PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    const viewMode = document.getElementById("dsViewMode").value;
    const isStacked = (viewMode === "stacked");
    const ratio = 1.0 - (p.y - MARGIN.top) / PLOT_HEIGHT;
    const curVal = ratio * currentDsYMax;
    const valStr = Math.round(curVal).toLocaleString() + (isStacked ? "%" : " 张");

    if (cursorBadge && cursorBadgeText && cursorBadgeRect) {{
      cursorBadge.style.display = "block";
      cursorBadgeText.textContent = valStr;

      const badgeW = Math.max(56, valStr.length * 7.5 + 14);
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

function hideDsCursorGuideline() {{
  const cursorGroup = document.getElementById("ds-cursor-group");
  if (cursorGroup) cursorGroup.style.display = "none";
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
  initTabs();
  initDatasetControls();
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
      vram_gb: parseFloat(r.e2e_vram_gb || 0)
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
      vram_gb: parseFloat(r.pat_vram_gb || 0)
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
      vram_gb: parseFloat(r.din_vram_gb || 0)
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
  updateCursorGuideline();
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

      rect.addEventListener("mouseenter", (e) => {{ showTooltip(e, item); updateCursorGuideline(e); }});
      rect.addEventListener("mousemove", (e) => {{ updateTooltipPos(e); updateCursorGuideline(e); }});
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
        }} else if (currentMetric === "vram_gb") {{
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
    text.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 22);
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
  xTitle.setAttribute("y", MARGIN.top + PLOT_HEIGHT + 45);
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
    <div class="tt-row"><span style="color:#93c5fd; font-weight:700;">【训练集自检与效能】</span></div>
    <div class="tt-row"><span>训练集良品自误报 (Clean FPR):</span> <span class="tt-val" style="color:${{p.clean_fpr > 1 ? '#f87171' : '#4ade80'}}; font-weight:600;">${{p.clean_fpr.toFixed(1)}}%</span></div>
    <div class="tt-row"><span>正常训练样本量:</span> <span class="tt-val" style="color:#fbbf24; font-weight:700;">N = ${{p.n}}</span></div>
    ${{NON_ITERATION_METRICS.includes(currentMetric) ? "" : `<div class="tt-row"><span>训练迭代轮次:</span> <span class="tt-val">${{p.iters.toLocaleString()}} 轮</span></div>`}}
    <div class="tt-row"><span>输入分辨率:</span> <span class="tt-val">${{sizeDesc}}</span></div>
    <div class="tt-row"><span>推理耗时 / FPS:</span> <span class="tt-val">${{p.lat_ms.toFixed(1)}} ms (${{Math.round(p.fps)}} FPS)</span></div>
    <div class="tt-row"><span>训练耗时 / 显存:</span> <span class="tt-val">${{p.train_time_m.toFixed(1)}} min | ${{p.vram_gb.toFixed(2)}} GB</span></div>
  `;
  tooltip.style.display = "block";
  updateTooltipPos(e);
}}

function updateTooltipPos(e) {{
  const chartBody = document.getElementById("chartBody");
  const rect = chartBody.getBoundingClientRect();
  let left = e.clientX - rect.left + 15;
  let top = e.clientY - rect.top + 15;

  if (left + 350 > rect.width) left -= 365;
  if (top + 340 > rect.height) top -= 345;

  tooltip.style.left = `${{left}}px`;
  tooltip.style.top = `${{top}}px`;
}}

function hideTooltip() {{
  tooltip.style.display = "none";
}}

let lastMousePos = null;

function getSvgCoords(e) {{
  const svg = document.getElementById("main-chart");
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
    x: ((e.clientX - rect.left) / rect.width) * SVG_WIDTH,
    y: ((e.clientY - rect.top) / rect.height) * SVG_HEIGHT
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

  const p = getSvgCoords(lastMousePos);
  if (!p) return;

  // Check if cursor is within vertical plot range and near chart horizontally
  const inPlotY = p.y >= MARGIN.top && p.y <= MARGIN.top + PLOT_HEIGHT;
  const inPlotX = p.x >= MARGIN.left - 45 && p.x <= MARGIN.left + PLOT_WIDTH + 30;

  if (inPlotY && inPlotX) {{
    cursorGroup.style.display = "block";

    // Horizontal dashed line across the plot width
    cursorHLine.setAttribute("x1", MARGIN.left - 4);
    cursorHLine.setAttribute("y1", p.y);
    cursorHLine.setAttribute("x2", MARGIN.left + PLOT_WIDTH);
    cursorHLine.setAttribute("y2", p.y);

    // Compute metric value corresponding to mouse Y position
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
  if (cursorGroup) {{
    cursorGroup.style.display = "none";
  }}
}}

function showDetailCard(p) {{
  const activeCfg = METRIC_CONFIG[currentMetric];
  const panel = document.getElementById("detailPanel");
  const sizeDesc = p.size === 672 ? "672 × 672 (斜线)" : (p.size === 448 ? "448 × 448 (横线)" : "224 × 224 (圆点)");
  panel.innerHTML = `
    <span><strong>选中实验:</strong> <span style="color: ${{p.color}}; font-weight: 700;">${{p.model}}</span> | <strong>${{activeCfg.label}}:</strong> <span style="color: #2563eb; font-weight: 700;">${{formatMetricVal(p.y_val, currentMetric)}}</span> | <strong>测试集AUROC:</strong> ${{p.auc.toFixed(4)}} | <strong>最优F1:</strong> ${{p.f1 ? p.f1.toFixed(4) : 'N/A'}} | <strong>测试集检出(TP)/漏检(FN)/误报(FP):</strong> <span style="color:#16a34a; font-weight:600;">${{p.tp}}</span> (${{p.recall.toFixed(1)}}%) / <span style="color:#dc2626; font-weight:700;">${{p.fn}}</span> (漏检: ${{p.fnr.toFixed(1)}}%) / <span style="color:#ea580c; font-weight:600;">${{p.fp}}</span> (误报: ${{p.fpr.toFixed(1)}}%) | <strong>训练集良品自误报:</strong> ${{p.clean_fpr.toFixed(1)}}% | <strong>N:</strong> ${{p.n}} ${{NON_ITERATION_METRICS.includes(currentMetric) ? "" : `| <strong>迭代:</strong> ${{p.iters.toLocaleString()}}`}} | <strong>尺寸:</strong> ${{sizeDesc}} | <strong>延迟:</strong> ${{p.lat_ms.toFixed(1)}}ms | <strong>耗时:</strong> ${{p.train_time_m.toFixed(1)}}min</span>
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
    const rect = chartBody.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    zoomAt(zoomFactor, mouseY);
  }}, {{ passive: false }});

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
  hideCursorGuideline();
  hideDsCursorGuideline();
  const isDsTab = (currentActiveTab === "dataset");
  const svgEl = isDsTab ? document.getElementById("dataset-chart") : document.getElementById("main-chart");
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svgEl);

  if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {{
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }}

  const blob = new Blob([source], {{ type: "image/svg+xml;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = isDsTab ? `dataset_split_distribution_${{DATASET_NAME}}.svg` : `${{currentMetric}}_vs_${{currentXDim}}_${{DATASET_NAME}}.svg`;
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


def get_all_dataset_split_stats(outs_dir: Path) -> Dict[str, Any]:
    """Extracts train sample sizes and test normal/defect counts for all datasets in base dir."""
    base_dir = outs_dir.parent
    stats = {}
    if not base_dir.is_dir():
        return stats
    for d in sorted(base_dir.iterdir()):
        sum_f = d / "final_multisize_summary.json"
        if d.is_dir() and sum_f.is_file():
            try:
                data = json.loads(sum_f.read_text(encoding="utf-8"))
                if not data:
                    continue
                item0 = data[0]
                ns = sorted(list(set(int(x.get("n", 0)) for x in data if "n" in x)))
                tp = int(item0.get("tp") or item0.get("din_tp", 0))
                fn = int(item0.get("fn") or item0.get("din_fn", 0))
                fp = int(item0.get("fp") or item0.get("din_fp", 0))
                tn = int(item0.get("tn") or item0.get("din_tn", 0))
                defect_cnt = tp + fn
                normal_cnt = fp + tn
                total_test = defect_cnt + normal_cnt

                stats[d.name] = {
                    "name": d.name,
                    "train_ns": ns,
                    "train_counts": {n: n for n in ns},
                    "test_defect": defect_cnt,
                    "test_normal": normal_cnt,
                    "test_total": total_test,
                    "is_current": (d.resolve() == outs_dir.resolve())
                }
            except Exception as e:
                print(f"[warn] Failed to parse stats for {d.name}: {e}")
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
    all_datasets_stats = get_all_dataset_split_stats(outs_dir)
    html_content = build_interactive_html(dataset_name, dataset_data, has_bank=has_bank, all_datasets_stats=all_datasets_stats)

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
