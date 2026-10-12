#!/usr/bin/env bash
# AI牧易通 · 端到端演示脚本
# 用法：先启动服务（uvicorn app.main:app --port 8000），再执行：
#   bash scripts/demo.sh
# 可用 BASE_URL 指定其他地址：BASE_URL=http://127.0.0.1:9000 bash scripts/demo.sh
# 登录：/api/* 需要令牌（默认账号 admin / admin123），脚本登录后自动携带。
set -u

BASE="${BASE_URL:-http://127.0.0.1:8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python || command -v python3 || true)"

say() { printf '\n\033[36m== %s ==\033[0m\n' "$1"; }
pretty() { if [ -n "$PY" ]; then "$PY" -m json.tool --no-ensure-ascii 2>/dev/null || cat; else cat; fi; }
py() { if [ -n "$PY" ]; then "$PY" -c "$1"; else cat > /dev/null; echo "(未找到 python，跳过格式化)"; fi; }

say "0. 健康检查（免登录）"
if ! curl -sf "$BASE/api/health" | pretty; then
  echo "服务未启动。请先运行：uvicorn app.main:app --port 8000"
  exit 1
fi

say "1. 登录（默认账号 admin / admin123，可在 .env 修改）"
TOKEN="$(curl -sf -X POST "$BASE/api/auth/login" -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  | py "import json,sys;print(json.load(sys.stdin)['data']['token'])")"
if [ -z "$TOKEN" ]; then
  echo "登录失败：请核对 .env 中的 AUTH_USERNAME / AUTH_PASSWORD"
  exit 1
fi
AUTH=(-H "Authorization: Bearer $TOKEN")
echo "登录成功，令牌有效期 12 小时（AUTH_TOKEN_TTL_SECONDS 可调）"

say "2. 单页驾驶舱（登录后进入：左侧菜单 12 页）"
curl -s -o /dev/null -w "GET /              -> HTTP %{http_code}（%{size_download} bytes）\n" "$BASE/"
curl -s -o /dev/null -w "GET /samples/xxx   -> 样例图挂载 HTTP %{http_code}\n" "$BASE/samples/sheep/P1-S-01.jpg"
curl -s -o /dev/null -w "GET /assets/xxx    -> 背景素材   HTTP %{http_code}\n" "$BASE/assets/bg-pasture.jpg"
curl -s -o /dev/null -w "GET /api/pastures  -> 未带令牌   HTTP %{http_code}（应为 401）\n" "$BASE/api/pastures"

say "3. 大屏概览（KPI）"
curl -sf "${AUTH[@]}" "$BASE/api/dashboard/overview" | pretty

say "4. 趋势与环境数据量（覆盖度逐草场独立，不做跨草场平均）"
curl -sf "${AUTH[@]}" "$BASE/api/dashboard/trends?days=30" | py "
import json, sys
d = json.load(sys.stdin)['data']
print(f\"草场 {len(d['pastures'])} 个，羊数序列 {len(d['sheep'])} 条\")
for p in d['pastures']:
    c = [x for x in p['coverage'] if x['coverage'] is not None]
    tail = f\"最新覆盖度 {c[-1]['coverage']:.2%}（{c[-1]['day']}）\" if c else '暂无覆盖度记录'
    print(f\"  {p['code']} {p['name']}：覆盖度序列 {len(p['coverage'])} 天，{tail}\")
if d['sheep']:
    print(f\"最新羊数 {d['sheep'][-1]['count']} 只\")
"
curl -sf "${AUTH[@]}" "$BASE/api/dashboard/environment?limit=48" | py "
import json, sys
r = json.load(sys.stdin)['data']['records']
print(f\"环境记录 {len(r)} 条，最新 {r[-1]['temperature']}℃ / 湿度 {r[-1]['humidity']}%\")
"

say "5. 羊群计数（上传编号样例 P2-S-01，YOLO11 检测；参考值 ≈149 只）"
curl -sf -X POST "${AUTH[@]}" "$BASE/api/vision/count" -F "file=@$ROOT/data/samples/sheep/P2-S-01.jpg" | py "
import json, sys
d = json.load(sys.stdin)['data']
print(f\"检测到 {d['count']} 只 {d['per_class']}，标注图 {len(d['annotated_image'])} bytes(base64)\")
"

say "6. 草地覆盖度（上传编号照片 P1-A-01，ExG+HSV）"
curl -sf -X POST "${AUTH[@]}" "$BASE/api/vision/coverage" -F "file=@$ROOT/data/samples/grassland/P1-A-01.jpg" | py "
import json, sys
d = json.load(sys.stdin)['data']
print(f\"覆盖度 {d['coverage']:.2%}，绿度 {d['greenness']:.3f}\")
"

say "7. 载畜量计算（NY/T 635，参数调用）"
curl -sf -X POST "${AUTH[@]}" "$BASE/api/vision/carrying-capacity" -H "Content-Type: application/json" \
  -d '{"yield_per_area":800,"area":100,"utilization":0.5,"grazing_days":30}' | pretty

say "8. AI Agent 思考流（SSE 实时事件：思考 → 工具调用 → 结果 → 决策）"
curl -sfN -X POST "${AUTH[@]}" "$BASE/api/agent/stream" -H "Content-Type: application/json" \
  -d '{"question":"东区草场现在草够吗？能放多少只羊？"}' | py "
import json, sys
for line in sys.stdin:
    if not line.startswith('data: '): continue
    e = json.loads(line[6:])
    if e['type'] == 'thinking': print('💭', e['text'][:64])
    elif e['type'] == 'tool_call': print('🔧 调用工具', e['name'])
    elif e['type'] == 'tool_result': print('📊', e['name'], '→', json.dumps(e['result'], ensure_ascii=False)[:72])
    elif e['type'] == 'answer':
        d = e['decision']
        print('✅ 决策：告警', d['alert_level'], '| 建议载畜量', d['recommended_stocking'], '羊单位')
        for a in d['recommended_actions']: print('   -', a)
    elif e['type'] == 'error': print('❌', e['message'])
"

say "9. 大屏决策流面板数据"
curl -sf "${AUTH[@]}" "$BASE/api/dashboard/decision-flow?limit=5" | py "
import json, sys
logs = json.load(sys.stdin)['data']['logs']
print(f'决策记录 {len(logs)} 条（最新在前）')
for l in logs[-3:]:
    print(f\" - [{l['created_at'][:16]}] {l['question'][:24]} | {l['alert_level']} | 工具 {len(l['tool_calls'])} 个\")
"

say "10. 系统配置信息（只读安全子集，不含密钥）"
curl -sf "${AUTH[@]}" "$BASE/api/system/info" | py "
import json, sys
d = json.load(sys.stdin)['data']
print(f\"阈值 {d['alert']['warning_threshold']}/{d['alert']['danger_threshold']} · 模型 {d['vision']['model_file']} · 大模型 {d['llm']['model']}（configured={d['llm']['configured']}）\")
print('数据存储:', ' · '.join(f\"{v['label']} {v['count']}\" for v in d['storage'].values()))
"

say "11. 账号管理（仅管理员；口令只存 PBKDF2 加盐哈希）"
curl -sf "${AUTH[@]}" "$BASE/api/users" | py "
import json, sys
d = json.load(sys.stdin)['data']
print(f\"账号 {len(d['users'])} 个：\", ' · '.join(f\"{u['username']}（{u['role_label']}）\" for u in d['users']))
print('角色字典:', d['roles'])
"

printf '\n\033[32m演示完成：打开 http://127.0.0.1:8000/ —— 先用 admin / admin123 登录，进入左侧分组菜单（12 个页面）\033[0m\n'
