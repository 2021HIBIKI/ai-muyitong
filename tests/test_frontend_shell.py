"""左侧分组菜单与多页外壳（阶段 9.4）：纯静态解析 ``static/index.html``。

护栏要点：
- 12 个页面（4 个旧视图 + 4 个能力独立页 + 巡检告警 + 3 个系统管理占位页）全部注册；
- 菜单分组与顺序固定；视图 DOM 顺序固定（test_api 的切片断言依赖 sheep < grass < agent）；
- 旧顶部按键（.tab/.tabs）已清除，快捷入口替代塔基卡片（塔基能力迁往独立页）。
"""
from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")

GROUPS = ["数据总览", "智能识别", "草场监测", "智能服务", "系统管理"]
VIEWS = ["overview", "sheep", "grass", "rectify", "zoning", "forecast",
         "health", "agent", "patrol", "devices", "users", "settings"]


def _index_of(marker: str) -> int:
    idx = HTML.index(marker)
    assert idx >= 0
    return idx


def test_sidebar_groups_and_menu_items():
    """五个分组标题齐全；每个菜单项都有 data-view 且指向已注册页面。"""
    for group in GROUPS:
        assert f'class="menu-group-title">{group}</div>' in HTML, group
    assert 'class="sidebar"' in HTML
    assert 'id="menu"' in HTML
    menu_items = HTML.split('id="menu"')[1].split("</nav>")[0]
    assert menu_items.count('class="menu-item') == len(VIEWS)
    for view in VIEWS:
        assert f'data-view="{view}"' in menu_items, view


def test_all_view_sections_exist_and_ordered():
    """12 个视图段落全部存在，且 DOM 顺序固定（旧断言依赖 sheep < grass < agent）。"""
    positions = [_index_of(f'id="view-{v}"') for v in VIEWS]
    assert positions == sorted(positions), "视图顺序必须与 VIEWS 列表一致"
    assert HTML.count('class="view') == len(VIEWS)


def test_page_header_and_user_block():
    assert '<header class="topbar">' in HTML
    assert 'id="pageTitle"' in HTML and 'id="pageSub"' in HTML
    assert 'class="topbar-right"' in HTML and 'id="clock"' in HTML
    assert 'class="user-block"' in HTML
    for anchor in ('id="userAvatar"', 'id="userName"', 'id="logoutBtn"'):
        assert anchor in HTML, anchor
    assert "已登录" in HTML


def test_view_registry_drives_hash_routing():
    """PAGES 注册表是唯一路由来源：switchView 查表，启动路由用 hasOwnProperty。"""
    assert "var PAGES = {" in HTML
    for view in VIEWS:
        assert re.search(rf"\b{view}:\s*\{{", HTML), f"PAGES 缺少 {view}"
    switch = re.search(r"function switchView\(name\) \{.*?\n  \}", HTML, re.S).group(0)
    assert "hasOwnProperty.call(PAGES, name)" in switch
    assert "pageTitle" in switch and "pageSub" in switch
    assert "menu-item" in switch
    start = re.search(r"function startApp\(\) \{.*?\n  \}", HTML, re.S).group(0)
    assert "hasOwnProperty.call(PAGES, initial)" in start


def test_menu_keyboard_and_press_feedback():
    """菜单键盘可达（bindRowKeyboard 覆盖 menu），按压反馈带 scale。"""
    assert 'bindRowKeyboard($("menu"))' in HTML
    assert re.search(r"\.menu-item:active \{[^}]*transform: scale\(", HTML)
    assert re.search(r"\.qe-btn:active \{[^}]*transform: scale\(", HTML)
    assert 'transform var(--dur-press) var(--ease-out)' in HTML  # 按压时长仍走 token


def test_hashchange_follows_registered_views():
    """地址栏改 hash / 浏览器后退前进 → 视图跟随（仅限已注册页面）。"""
    assert 'addEventListener("hashchange"' in HTML
    assert "hasOwnProperty.call(PAGES, name)" in HTML.split('addEventListener("hashchange"')[1][:400]


def test_legacy_tab_class_removed():
    """旧顶部按键已清除：不再有 .tabs/.tab 结构与样式。"""
    assert 'class="tab' not in HTML
    assert ".tabs {" not in HTML
    assert 'id="tabs"' not in HTML


def test_quick_entry_replaces_tower_card_in_grass_view():
    """草场分析页：塔基卡片移除，四个快捷入口按钮在位；塔基交互锚点迁出草场页。"""
    grass_view = HTML.split('id="view-grass"')[1].split('id="view-rectify"')[0]
    assert 'id="quickEntry"' in grass_view
    assert grass_view.count('class="qe-btn"') == 4
    for gone in ('id="towerDrop"', 'id="towerCanvas"', 'id="towerRunBtn"',
                 'id="towerPointHint"', 'id="towerModeRow"', 'id="zoningRunBtn"'):
        assert gone not in grass_view, gone


def test_footline_and_toast_host_remain_in_stage():
    """底部状态行与告警弹窗容器仍在舞台内（弹窗定位依赖 stage 坐标系）。"""
    assert 'id="toastHost"' in HTML
    assert 'class="footline"' in HTML
    assert 'id="lastUpdate"' in HTML


def test_overview_renamed_and_views_scrollable():
    """命名：菜单/页头/注册表统一为「总览界面」（不再出现总览驾驶舱）；
    视图容器溢出时可滚动（低高度窗口下内容不被裁掉且可滚到）。"""
    assert "总览界面" in HTML
    assert "总览驾驶舱" not in HTML
    assert ".view { position: absolute; inset: 0; display: none; padding-top: 18px; overflow-y: auto; }" in HTML
