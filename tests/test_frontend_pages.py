"""四个能力独立页的前端护栏（阶段 9.5+）：纯静态解析 + 真实算法交叉验证。

本文件按页面分节，随各页实现逐步扩充。当前覆盖：
- 塔基几何校正页：标记齐全、画布 620×420、调用 /api/vision/orthorectify、
  单应矩阵展示；**样例四边形与资产交叉验证**（用前端硬编码的 4 点对样例
  资产跑真实 orthorectify：重投影误差≈0 且校正后覆盖度恢复到源图真值）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

HTML = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def _view_slice(name: str) -> str:
    views = [m.group(1) for m in re.finditer(r'id="view-(\w+)"', HTML)]
    assert name in views, f"视图缺失：view-{name}"
    start = HTML.index(f'id="view-{name}"')
    end = len(HTML)
    for other in views[views.index(name) + 1 :]:
        end = min(end, HTML.index(f'id="view-{other}"'))
    return HTML[start:end]


def _parse_tower_sample_quad() -> list[list[float]]:
    m = re.search(r"var TOWER_SAMPLE_QUAD = (\[\[.*?\]\]);", HTML)
    assert m, "前端缺少 TOWER_SAMPLE_QUAD"
    quad = json.loads(m.group(1))
    assert len(quad) == 4 and all(len(p) == 2 for p in quad)
    assert all(0 <= v <= 1 for p in quad for v in p)
    return quad


# ---------------------------------------------------------------------------
# 塔基几何校正独立页
# ---------------------------------------------------------------------------

def test_rectify_view_markup():
    view = _view_slice("rectify")
    for anchor in ('id="rectifyDrop"', 'id="rectifyCanvas"', 'id="rectifySampleBtn"',
                   'id="rectifyRunBtn"', 'id="rectifyResetBtn"', 'id="rectifyPointHint"',
                   'id="rectifyResult"', 'id="rectifyMeta"'):
        assert anchor in view, anchor
    # 步骤指示：载入 → 点选 → 校正
    for step in ("rectStep1", "rectStep2", "rectStep3"):
        assert f'id="{step}"' in view, step
    assert "按 左上 → 右上 → 右下 → 左下 顺序点选" in view


def test_rectify_and_zoning_principle_cards_removed():
    """校正/分区页的独立「原理」大卡已移除（腾出展示空间）；
    校正页「思路来源」与分区页「轻量实现」两处来源小字随后按反馈一并移除，
    论文出处仅在 README/报告书中标注（界面保持干净）。"""
    for view_name in ("rectify", "zoning"):
        seg = _view_slice(view_name)
        assert '<div class="card-title">原理' not in seg, view_name
        assert "思路来源" not in seg and "覃星力" not in seg and "轻量实现" not in seg, view_name


def test_rectify_canvas_840x540():
    """画布内部分辨率 840×540（CSS 铺满容器、object-fit 保比例）。"""
    assert '("rectifyCanvas").width = 840' in HTML
    assert '("rectifyCanvas").height = 540' in HTML
    assert "#rectifyCanvas { width: 100%; height: 100%; object-fit: contain" in HTML


def test_rectify_posts_orthorectify_with_points():
    """校正按钮调用 /api/vision/orthorectify：multipart 图片 + 4 点 JSON。"""
    assert '"/api/vision/orthorectify"' in HTML
    assert 'fd.append("points", JSON.stringify(towerState.points))' in HTML
    assert 'fd.append("file", towerState.file' in HTML


def test_rectify_shows_matrix_and_compare():
    """结果卡：H 矩阵（3×3 等宽）、校正前后覆盖度对比条。"""
    assert 'class="h-matrix"' in HTML
    assert "matrixText" in HTML and "单应矩阵 H" in HTML
    assert "bar-fill before" in HTML and "bar-fill after" in HTML
    assert "reprojection_error" in HTML  # 重投影误差徽标


def test_rectify_sample_button_prefills_quad():
    """「载入样例」自动填入内置四边形并启用校正按钮（一键演示）；
    样例 URL 带 ?v= 版本号（重新生成样例后绕开浏览器缓存的守卫）。"""
    sample_flow = HTML.split('$("rectifySampleBtn").addEventListener')[1].split(
        '}).catch(function (e) { alert("样例载入失败'
    )[0]
    assert "TOWER_SAMPLE_QUAD.map" in sample_flow
    assert "rectifyRunBtn" in sample_flow and "disabled = false" in sample_flow
    assert 'assets/samples/tower_oblique.jpg?v=' in HTML


def test_tower_sample_quad_rectifies_asset_to_source_truth():
    """交叉验证：前端硬编码四边形对样例资产做真实校正——
    重投影误差≈0，校正后覆盖度恢复源图真值（ICAERUS 牧地裁剪区，±0.05），
    且显著高于斜拍直测（透视压缩低估的演示故事成立）。"""
    from app.vision.georectify import orthorectify, reprojection_error
    from app.vision.grassland import coverage

    quad = _parse_tower_sample_quad()
    img = cv2.imread(str(ROOT / "static" / "assets" / "samples" / "tower_oblique.jpg"))
    assert img is not None

    result = orthorectify(img, quad)
    error = reprojection_error(result["matrix"], result["src_points"], result["dst_points"])
    # 逆变换为最小二乘解，浮点残差 ~1e-5 px；API round(4) 后即 0.0 px（演示口径）
    assert error < 0.01, f"样例四边形重投影误差应≈0：{error}"

    before = coverage(img)["coverage"]
    after = coverage(result["image"])["coverage"]
    assert after > before + 0.2, f"校正应显著提升覆盖度：{before:.3f} → {after:.3f}"

    # 真值 = 生成脚本的源裁剪区（scripts/make_tower_sample.py: CROP = 0,700,4000,3000）
    source = cv2.imread(str(ROOT / "data" / "samples" / "mauron_example1.JPG"))
    truth = coverage(source[700:3000, 0:4000])["coverage"]
    assert abs(after - truth) < 0.05, f"校正后应恢复源图真值 {truth:.3f}：{after:.3f}"


# ---------------------------------------------------------------------------
# 分区自动分割独立页
# ---------------------------------------------------------------------------

def test_zoning_view_markup_and_controls_2_to_6():
    view = _view_slice("zoning")
    for anchor in ('id="zoningDrop"', 'id="zoningFile"', 'id="zoningSampleBtn"',
                   'id="zoningRunBtn"', 'id="zoningPlots"', 'id="zoningPlotsVal"',
                   'id="zoningResult"', 'id="zoningMeta"'):
        assert anchor in view, anchor
    # 分区数量滑杆：2–6（后端 1–9，前端收敛到管理口径）
    assert 'id="zoningPlots" min="2" max="6" step="1" value="3"' in view
    assert "覃星力" not in view  # 来源小字已按反馈移除（出处在 README/报告书）


def test_zoning_posts_segment_with_n_plots():
    assert '"/api/vision/segment"' in HTML
    assert 'fd.append("n_plots", $("zoningPlots").value)' in HTML
    assert 'fd.append("file", zoningState.file' in HTML


def test_zoning_renders_zone_cards_and_mask_ratio():
    """结果卡：彩色叠加图 + 分区卡片（覆盖度/面积占比，前端按 area 归一）+
    全图植被占比与连通域数；连通域不足目标分区数时如实说明。"""
    assert 'class="zone-grid"' in HTML and 'class="zone-card"' in HTML
    assert "全图植被占比" in HTML
    assert "pctText(d.mask_ratio)" in HTML
    assert "reduce(function (s, p) { return s + (p.area || 0); }" in HTML  # 面积占比自算
    assert "d.n_components" in HTML
    assert "annotated_image" in HTML
    assert "连通域不足目标分区数" in HTML  # 分区数受连通域约束时如实说明


def test_zoning_sample_uses_real_aerial_pasture():
    """分区样例用真实无人机正俯拍牧地（ICAERUS mauron_example1，/samples 挂载），
    而非合成图或地面特写。"""
    sample_flow = HTML.split('$("zoningSampleBtn").addEventListener')[1].split(".catch(")[0]
    assert "/samples/mauron_example1.JPG" in sample_flow


# ---------------------------------------------------------------------------
# 覆盖度预测独立页
# ---------------------------------------------------------------------------

def test_forecast_view_markup_and_horizons():
    view = _view_slice("forecast")
    for anchor in ('id="fcPastures"', 'id="fcHorizon"', 'id="forecastChart"',
                   'id="fcCurrent"', 'id="fcTail"', 'id="fcTrend"', 'id="fcResidual"',
                   'id="fcRisk"', 'id="fcValues"', 'id="fcSub"'):
        assert anchor in view, anchor
    for days in ("3", "7", "14", "30"):
        assert f'data-days="{days}"' in view, days


def test_forecast_fetches_trend_with_horizon_and_pasture_selector():
    assert '"/trend?days=30&forecast_days=" + fcState.horizon' in HTML
    assert 'apiJSON("/api/pastures")' in HTML  # 草场选择器由接口清单驱动
    assert "data-fc-pasture" in HTML
    assert "PAGES.forecast.onShow" in HTML  # 惰性初始化在进入页面时执行


def test_forecast_side_cards_and_risk_hint():
    """指标卡：当前值/预测期末值/日均变化/残差标准差；风险提示触及阈值才出现
    且声明「不触发告警判定」（趋势预警与告警口径分离）。"""
    for anchor in ('fcCurrent', 'fcTail', 'fcTrend', 'fcResidual'):
        assert f'$("{anchor}").textContent' in HTML, anchor
    assert "fc.residual_std" in HTML and "fc.trend" in HTML
    assert "触及 35% 危险线" in HTML and "低于 60% 正常线" in HTML
    assert "不触发告警判定" in HTML


def test_trend_band_logic_shared_between_grass_and_forecast():
    """置信带逻辑只存在一份（buildTrendOption 抽取），草场详情与预测页共用；
    草场详情的旧调用点（forecast_days=7）保持不动。"""
    assert "function buildTrendOption(" in HTML
    assert 'buildTrendOption(data, { compact: true })' in HTML   # 草场详情（紧凑）
    assert 'buildTrendOption(data, { compact: false })' in HTML  # 预测页（完整）
    assert HTML.count("置信上界") == 1  # 无重复实现（去重护栏）
    assert '"forecast_days=7"' in HTML or '"/trend?days=30&forecast_days=7"' in HTML


# ---------------------------------------------------------------------------
# 多模态评分独立页
# ---------------------------------------------------------------------------

def test_health_view_markup_and_weights():
    view = _view_slice("health")
    for anchor in ('id="healthGrid"', 'id="healthDetail"', 'id="healthDetailSub"',
                   'id="healthTemp"', 'id="healthHumidity"', 'id="healthCompareChart"'):
        assert anchor in view, anchor
    # 权重口径与后端 grass_health_score 一致（覆盖度 60 / 绿度 15 / 环境 25）
    assert "HEALTH_WEIGHTS = { coverage: 0.6, greenness: 0.15, environment: 0.25 }" in HTML
    assert "覆盖度 60%" in view and "绿度 15%" in view and "环境 25%" in view
    # 权重说明全文可滚动查看（窄高度下不被截断）
    assert 'id="healthWeightNote"' in view and "overflow-y:auto" in view.split('id="healthWeightNote"')[1][:120]


def test_health_cards_render_score_level_and_three_bars():
    """评分卡：大分数 + 等级 chip + 三条贡献条（缺失显示「缺失」不虚构）。"""
    assert "'<div class=\"hcard'" in HTML  # JS 拼接的卡片模板
    assert 'class="hcard-score"' in HTML
    assert "hbar" in HTML
    assert "缺失" in HTML
    assert "ALERT_LABEL[level]" in HTML
    assert "PAGES.health.onShow" in HTML


def test_health_detail_reads_missing_and_renormalized_weights():
    """贡献分解：真实展示缺失模态与归一化后权重（不显示虚假的固定权重）。"""
    assert "h.missing" in HTML
    assert "归一化后" in HTML and "已按剩余权重归一化" in HTML
    assert "不虚构数据" in HTML
    assert "HEALTH_WEIGHTS[k] / activeWeight" in HTML


def test_health_env_from_environment_endpoint():
    """当前环境卡读 /api/dashboard/environment（温度/湿度真实数据）。"""
    assert '"/api/dashboard/environment?limit=1"' in HTML
    assert '("healthTemp").textContent' in HTML and '("healthHumidity").textContent' in HTML


def test_health_compare_chart_colored_by_level():
    """评分对比柱状图按告警等级着色（正常绿/预警橙/危险红）。"""
    assert "HEALTH_COLORS = { normal:" in HTML
    assert "itemStyle" in HTML and "params.dataIndex" in HTML


# ---------------------------------------------------------------------------
# 巡检告警独立页 + 系统管理组（设备台账 / 用户管理 / 系统设置）
# ---------------------------------------------------------------------------

def test_patrol_view_markup_and_shared_data_source():
    """巡检页：立即巡检 + 全量告警表（与概览面板同一 loadAlerts 数据源）。"""
    view = _view_slice("patrol")
    for anchor in ('id="patrolPageRunBtn"', 'id="patrolPageResult"', 'id="patrolAlertList"'):
        assert anchor in view, anchor
    load = re.search(r"function loadAlerts\(\) \{.*?\n  \}", HTML, re.S).group(0)
    assert "renderAlerts(list)" in load and "renderPatrolTable(list)" in load
    assert "PAGES.patrol.onShow" in HTML


def test_patrol_table_has_ack_buttons():
    """全量表格行：等级/标题/草场覆盖度/时间/已知悉；确认后按钮变灰
    （disabled 的「已知悉」，不显示「已已知悉」等怪文案）；确认后刷新两处面板。"""
    table = re.search(r"function renderPatrolTable\(list\) \{.*?\n  \}", HTML, re.S).group(0)
    assert "data-ack" in table
    assert 'disabled>已知悉</button>' in table  # 已确认 → 灰态
    assert "已已知悉" not in HTML
    assert "pctText(a.metrics.coverage)" in table
    assert 'acknowledgeAlert(Number(btn.dataset.ack)' in HTML
    # 灰态样式：不透明蓝色改为中性灰
    assert ".mini-btn:disabled { opacity: 1; color: var(--ink-3);" in HTML


def test_patrol_run_result_shown_on_both_surfaces():
    """手动巡检结果同步渲染：概览面板（汇总头）与巡检页（紧凑行）。"""
    fn = re.search(r"function renderPatrolResult\(data\) \{.*?\n  \}", HTML, re.S).group(0)
    assert '("patrolResult")' in fn and '("patrolPageResult")' in fn
    assert "patrolRowsHtml(results)" in fn


def test_devices_page_honest_labels():
    """设备台账如实标注：硬件全部「待接入」，不虚构已接入/在线；
    仅云端平台（本项目软件本身）为「运行中」；「本地快速检测/云端深度分析」
    分层架构文字留在设备用途中（独立「说明」卡已按反馈移除，
    状态与架构口径改由 DEVICES 数据守卫）。"""
    view = _view_slice("devices")
    assert "规划阶段" in view  # 选型台账定位（与报告书硬件章节同口径）
    assert "未接入" not in view
    assert '<div class="card-title">说明' not in view  # 独立说明卡已移除
    assert "运行中" in HTML  # 云端平台状态（DEVICES 数据中，仅此一条）
    # 硬件行全部待接入：DEVICES 中仅云端平台 ok: true；分层架构文字在设备用途中
    devices = HTML.split("var DEVICES = [")[1].split("];")[0]
    assert devices.count('status: "待接入"') == 5
    assert devices.count("ok: false") == 5
    assert devices.count("ok: true") == 1
    assert "本地快速检测" in devices and "云端深度分析" in devices


def test_users_page_implements_account_management():
    """用户管理：当前账号（/api/auth/me 含角色）+ 账号列表（/api/users）+
    新增账号表单（用户名/口令/角色）+ 管理员专属操作（重置密码/删除）。"""
    view = _view_slice("users")
    for anchor in ('id="usersName"', 'id="usersAvatar"', 'id="usersRoleTag"',
                   'id="userList"', 'id="userListMeta"', 'id="userCreateCard"',
                   'id="nuName"', 'id="nuPass"', 'id="nuRole"', 'id="nuCreateBtn"'):
        assert anchor in view, anchor
    assert '"/api/auth/me"' in HTML
    assert '"/api/users"' in HTML
    assert '"/password"' in HTML  # 重置密码
    # 角色与诚实说明
    assert 'ROLE_LABEL = { admin: "管理员", herder: "放牧员", ranger: "巡护员" }' in HTML
    assert "PBKDF2" in view and "RBAC" in view
    assert "服务端不做吊销" in view  # 无状态令牌如实说明
    # 非管理员降级：隐藏新增卡 + 只读提示
    assert "仅管理员可操作" in view or 'role === "admin" ? "flex" : "none"' in HTML


def test_settings_page_renders_system_info():
    """系统设置：三组配置行 + 数据存储统计由 /api/system/info 驱动，明示不含密钥。"""
    view = _view_slice("settings")
    for anchor in ('id="setApp"', 'id="setVision"', 'id="setLLM"', 'id="setStorage"'):
        assert anchor in view, anchor
    assert '"/api/system/info"' in HTML
    assert '("setApp").innerHTML' in HTML
    assert '("setStorage").innerHTML' in HTML
    assert "数据存储" in view and "SQLite" in view
    assert "不暴露任何密钥" in view


# ---------------------------------------------------------------------------
# 羊群识别快速样例：编号即归属（P1-S-01…），与 data/samples/sheep/manifest.json 一致
# ---------------------------------------------------------------------------

def test_sheep_sample_buttons_numbered_and_match_manifest():
    """快速样例按钮 10 个：编号=草场-用途-序号（P1-S-01…），data-src 指向
    /samples/sheep/<编号>.jpg，按钮文案含编号与清单中的参考计数——
    前端按钮与样例清单（含生成时验证的检出）双向一致，防止改名/换图后失联。"""
    block = HTML.split('id="sheepSamplesImage"')[1].split('id="sheepSamplesVideo"')[0]
    assert "草场-用途-序号" in block and "S 为航拍羊群" in block  # 编号口径声明
    buttons = re.findall(r'data-src="([^"]+)">([^<]+)</button>', block)
    assert len(buttons) == 10

    manifest = json.loads(
        (ROOT / "data" / "samples" / "sheep" / "manifest.json").read_text(encoding="utf-8")
    )
    by_code = {s["code"]: s for s in manifest["samples"]}
    assert len(by_code) == 10
    used = set()
    for src, label in buttons:
        m = re.fullmatch(r"/samples/sheep/(P\d+-S-\d{2})\.jpg", src)
        assert m, f"样例 URL 应为 /samples/sheep/<编号>.jpg：{src}"
        code = m.group(1)
        used.add(code)
        assert code in label, f"按钮文案应含编号：{label}"
        truth = by_code[code]["truth"]
        if truth:
            assert f"≈{truth}只" in label, f"参考计数应与清单一致：{label} vs {truth}"
        else:
            assert "0只" in label, label  # 空场景不带 ≈
    assert used == set(by_code), "按钮与清单必须一一对应"


# ---------------------------------------------------------------------------
# 布局护栏：新页面内容吃满视图高度（用户反馈修复：页面未铺满）
# ---------------------------------------------------------------------------

def test_new_pages_fill_viewport_height():
    """八个新页面每页主内容均吃满高度；关键容器带 flex:1 / min-height:0；
    正文卡不得再退回 flex:none（自然高度会在视图底部留空白）。"""
    # 全局机制：.split 吃满视图剩余高度
    assert ".split { flex: 1; min-height: 0;" in HTML
    for view in ("rectify", "zoning", "forecast", "health", "patrol", "devices", "users"):
        segment = _view_slice(view)
        assert "flex:1" in segment or "flex: 1" in segment, view
    # 逐页关键锚点（内联样式钉死；设置页用 2×2 网格行均分，无需 flex 字面量）
    assert 'id="rectifyStage" style="display:none;flex:1;flex-direction:column;min-height:0"' in HTML
    assert 'id="forecastChart" style="flex:1;min-height:320px"' in HTML
    assert 'id="healthCompareChart" style="flex:1;min-height:120px"' in HTML
    assert 'id="zoningResult" style="display:none;flex-direction:column;gap:14px;flex:1;min-height:0"' in HTML
    assert 'id="patrolAlertList" class="palert-list"' in HTML
    # 关键 CSS 规则
    assert ".palert-list { display: flex; flex-direction: column; gap: 8px; flex: 1;" in HTML
    assert ".fc-values { display: flex; flex-direction: column; gap: 4px; flex: 1;" in HTML
    assert ".annotated { width: 100%; flex: 1;" in HTML
    assert ".tower-result img { width: 100%; flex: 1;" in HTML
    assert "background: rgba(255, 255, 255, 0.6); flex: 1; }" in HTML  # .device-row 均分高度
    assert "grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr" in HTML  # 设置页 2×2 撑满