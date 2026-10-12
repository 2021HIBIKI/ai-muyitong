# AI牧易通 · 数字牧业全链平台（MVP）

FastAPI 单体应用：羊群计数（YOLO11 羊专用权重 + 跟踪去重）、草地覆盖度（ExG+HSV 植被分割）、载畜量计算（NY/T 635）、AI Agent（GLM 工具调用 + 本地规则降级）、ECharts 数据大屏与后台管理。

## 环境安装

### 方式 A：CPU / 通用环境（无 GPU 要求）

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方式 B：AMD 显卡 GPU 加速（Windows ROCm SDK）

适用已按「Windows 版 PyTorch-ROCm」方案部署过 ROCm SDK 的机器（`torch` 需为 `+rocmsdk` 构建、Python 3.12）。

1. 若已有可用的 conda 环境（如 ComfyUI 的 `comfy` 环境），直接克隆，不污染原环境：

   ```bash
   conda create -n muyitong --clone comfy -y
   ```

2. 装入本项目依赖（requirements.txt 不含 torch，不会替换 ROCm 版 torch）：

   ```bash
   conda activate muyitong
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 验证：`python -c "import torch; print(torch.cuda.is_available())"` 输出 `True` 即 GPU 可用。`VISION_DEVICE=auto` 下自动使用 GPU（ROCm SDK 的 torch 同样报告 cuda 可用）。模型加载后自动做一次空图预热，避免 HIP 内核冷编译导致首次推理为空。

### 视频跟踪依赖（阶段 6，务必保护 ROCm 环境）

`supervision`（ByteTrack / LineZone）及其运行期依赖按 **--no-deps** 安装，避免 pip 解析依赖时升级 numpy —— ROCm 版 torch 对 numpy 版本敏感，升级会导致 GPU 推理失效：

```bash
pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple supervision==0.30.3 "pydeprecate<0.12" defusedxml
# scipy 若缺失：pip install scipy（纯 wheel，不会动 numpy）
```

装完核对 `numpy` 版本与安装前一致、`torch.cuda.is_available()` 仍为 `True`。supervision 声明的 `av` / `matplotlib` / `requests` 本项目不使用（只做检测帧输入 + 帧序列输出）。

## 配置

复制 `.env.example` 为 `.env`（已 gitignore）：

- `GLM_API_KEY`：智谱开放平台 API Key（[open.bigmodel.cn](https://open.bigmodel.cn/)）；留空则 Agent 走本地关键字规则降级，全部功能仍可用。
- `GLM_MODEL` 默认 `glm-4.7-flash`；`LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF` 控制瞬时错误（429/5xx/超时）指数退避重试。
- `AUTH_USERNAME` / `AUTH_PASSWORD`：**首次启动播种的管理员账号**（默认 `admin` / `admin123`，演示开箱即用；**正式部署必须修改**）。账号存于 `users` 表，之后在「用户管理」页维护（新增账号 / 重置密码 / 删除）；改 .env 口令仅在账号不存在时生效。
- `AUTH_TOKEN_SECRET`：令牌签名密钥（HMAC-SHA256）；改动后所有已签发令牌立即失效，**正式部署必须修改**。
- `AUTH_TOKEN_TTL_SECONDS`：令牌有效期（秒，默认 43200 = 12 小时）。
- `VISION_MODEL_PATH`：羊专用权重路径（默认 `models/yolo11n_all.pt`，不存在时自动用 COCO 兜底权重 `yolov8s.pt`）。
- `VISION_DEVICE`：`auto`（优先 CUDA/ROCm，否则 CPU）| `cuda` | `cpu`。
- `VISION_MAX_DET`：单帧最大检测数（默认 600；ultralytics 默认 300 会把航拍大群截断）。
- 视频计数：`VIDEO_MAX_BYTES` / `VIDEO_MAX_SECONDS` / `VIDEO_STRIDE`（检测步长，默认 2）/ `VIDEO_MAX_PREVIEWS`；越线线位 `LINE_X1..LINE_Y2`（归一化，默认自右向左的 y=40% 横线：走向镜头记为 inbound）。
- 主动巡检：`PATROL_ENABLED`（默认 false，开启后后台定时巡检）、`PATROL_INTERVAL_SECONDS`（默认 60）、`PATROL_COOLDOWN_SECONDS`（同场同级告警冷却，默认 300）、`PATROL_IMAGE_PATH`（巡检点位图片，仅在库中无草场时作为单点位兜底；有草场时逐草场用各自编号照片巡检）。手动触发始终可用：`POST /api/patrol/run`。

### 权重放置

两组羊专用权重都放在 `models/`：`yolo11n_all.pt`（公开权重，默认部署）与 `yolo11n_private.pt`（私有数据集微调权重，可选部署，见下节）。二者都不进本开发仓库（`*.pt` 已 gitignore，自行放入 `models/` 即可），但**源码提交包内随包提供**；任一缺失时自动降级为 COCO 权重 `yolov8s.pt`（首次使用会自动下载）。

### 私有数据集与模型微调（复现报告书表 3-2）

除默认部署的公开羊专用权重外，本产品还提供一组由本团队训练的权重：以公开无人机羊群影像（[Mendeley Data](https://data.mendeley.com/datasets/nppwbkdf85/1)，原始包 >5GB 需自行下载）为源，经确定性管线处理成私有数据集 **Muyitong-Sheep**，再在其上微调 YOLO11n 得到。四步复现：

```bash
python scripts/make_private_dataset.py   # 1. 建私有数据集：质量筛除→标注转换→密集切片→统一分辨率→增强×4→分层划分（seed=42）
python scripts/train_private_ft.py       # 2. GPU 微调 30 轮（约 8 分钟；无 GPU 亦可跑，慢十余倍）
python scripts/eval_private_dataset.py   # 3. 私有验证集评测：公开权重 vs 微调权重 → data/private_dataset/eval_metrics.json
python scripts/compare_weights.py        # 4. 两组权重在演示资产上的对比 → data/private_dataset/weight_comparison.json
```

实测（28 张验证图 / 663 只羊，同一推理设置）：mAP@0.5 **29.9% → 93.2%**，精确率 45.9% → 90.8%，召回率 38.9% → 86.6%。

**部署哪一个权重**：两组都随包提供，**默认部署公开权重** `models/yolo11n_all.pt`。对比实测见 `scripts/compare_weights.py`（结果存 `data/private_dataset/weight_comparison.json`）：微调权重在与训练域同源的无人机俯拍样例上更准（平均计数误差 1.1 只 vs 2.4 只），但在异源测试图库（ICAERUS 公路/地面视角 74 张）上系统性低估（中位 −13 只，59 张偏差超 10%；公开权重中位偏差 0）——30 轮微调把有效域收窄到了俯拍放牧视角。要面向塔基俯拍场景部署，改 `VISION_MODEL_PATH=models/yolo11n_private.pt` 即可切换。

数据集图像/标签（`data/private_dataset/images|labels/`）、训练产物（`runs/`）与原始数据包均不入库；入库的是**可复算**的部分：数据清单 `manifest.json`（每张源图的处理轨迹与统计）、数据集配置 `dataset.yaml`、评测结果 `eval_metrics.json` 与权重对比 `weight_comparison.json`、四个管线脚本。

## 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 http://127.0.0.1:8000/ —— **先登录**（默认 `admin` / `admin123`，`.env` 可改；令牌存本地，12 小时有效），进入**左侧分组菜单 12 页**驾驶舱：

- **总览驾驶舱**：牧场状态、4 项 KPI（监测草场数 / 告警等级 / 建议载畜量 / 羊群数量，**各草场独立**：等级取各场中最严重、载畜量取各场独立核定后之和）、近 48 小时温湿度时序（1 小时/点，可缩放）、**巡检告警面板**（逐场巡检结果 + SSE 实时弹出 + 5 秒轮询兜底 + 手动「立即巡检」+ 已知悉；点击实时弹窗跳转并高亮面板）、Agent 决策流（**点击任一条跳转到 AI 助手回放该次问答**）。标题下为**各草场逐场状态**（编号 · 覆盖度 · 等级，点击进入该草场分析）
- **羊群识别**：**图片计数**（拖拽/样例图上传 → YOLO11 计数 + 检测框标注图）与**视频计数**（上传视频或选内置素材 → 逐帧检测 + ByteTrack 跟踪 + 越线进出统计 → 标注帧序列逐帧播放、数字同步跳动）。图片计数内置 10 个**编号快速样例**（编号即归属：草场-用途-序号，S=航拍羊群照片，如 `P1-S-01`；素材为 Mendeley 真实无人机 Merino 羊数据集，`data/samples/sheep/`，含标注参考计数清单）；完整测试图库（74 张，含检出数清单 `manifest.json`）在 `data/samples/gallery/`；视频素材（2 条 Pexels）在 `data/samples/video/`
- **草场分析**：多草场管理（新建 / 切换 / 删除）——一个草场由多个分区的编号照片（草场-分区-序号，如 `P1-A-01`）组成；一键分析全部照片 → 各分区与**本草场**覆盖度判定（草场内部聚合，草场之间彼此独立不做平均）；草场按日趋势与载畜量建议（按草场面积自动带入）。内置 3 个演示草场共 10 张编号照片（`data/samples/grassland/`，覆盖 正常/预警/危险 三档草情）；页面右侧为四个能力独立页的**快捷入口**
- **塔基几何校正**（独立页）：塔基斜拍图 → 四点标定 → 单应变换正射校正；步骤指示 + 画布点选 + **「载入样例」一键演示**（内置斜拍模拟图与地面四边形——源为 ICAERUS 真实无人机俯拍牧地，斜拍直测 42.1% → 校正后 88.4%，恢复源图真值 86.4%）；结果卡展示重投影误差、校正前后覆盖度/绿度对比与 H 矩阵（3×3）
- **分区自动分割**（独立页）：草场图 → 植被掩膜连通域聚类 → A/B/C… 管理分区；分区数量滑杆（2–6）、彩色叠加图、分区卡片（覆盖度/面积占比）、全图植被占比与连通域数（连通域不足目标分区数时如实说明）
- **覆盖度预测**（独立页）：草场选择 + 预测步长（3/7/14/30 天）、大趋势图（实测 + 预测虚线 + 置信带）、指标卡（当前/期末/日均变化/残差标准差）、风险提示（预测触及 60%/35% 阈值时分级提示，**不触发告警判定**）
- **多模态评分**（独立页）：各草场评分卡（覆盖度 60% · 绿度 15% · 环境 25%，缺失自动归一）、贡献分解（含归一化后权重如实标注）、当前环境温湿度、跨草场评分对比柱状图（按告警等级着色）
- **AI 助手**：自然语言提问 → **大模型思考过程实时滚动**（SSE：思考/工具调用/结果/确定性分析，带耗时戳）→ 结构化决策卡；**历史对话**（每次问答含思考过程自动落库，点击任一条可回放；概览决策流点击跳转到这里）。可直接问具体草场（如「北坡轮休草场草情如何？」），数字与草场分析视图同口径
- **巡检告警**（独立页）：全量告警历史表（等级/标题/覆盖度/时间/已知悉）+「立即巡检」逐场结果（与概览面板同一数据源）
- **设备台账 / 用户管理 / 系统设置**：硬件方案选型台账（如实标注「待接入」）与云端平台状态；**用户管理**（账号列表 + 新增账号 + 重置密码 + 删除，角色三档：管理员/放牧员/巡护员，仅管理员可维护账号）；系统设置（只读运行时配置 + 数据存储统计，`GET /api/system/info`，不含密钥）

（业务表为空时启动自动播种 30 天模拟数据，开箱即有数据可演示。）

## 接口

| 接口 | 说明 |
|---|---|
| `POST /api/auth/login` | 登录（账号存于 users 表，口令 PBKDF2 加盐哈希校验 → 无状态令牌；失败返回 `code=401` 统一信封） |
| `GET /api/auth/me` | 当前登录账号（用户名 + 角色；账号被删除时 HTTP 401） |
| `GET /api/users` | 账号列表（**仅管理员**；含角色字典，绝不回显哈希） |
| `POST /api/users` | 新建账号（仅管理员；用户名唯一、口令 ≥6 位、角色三选一） |
| `DELETE /api/users/{id}` | 删除账号（仅管理员；不能删自己、至少保留一个管理员） |
| `POST /api/users/{id}/password` | 重置账号口令（仅管理员） |
| `GET /api/system/info` | 运行时配置安全子集 + 数据存储统计（**不含任何密钥与路径**） |
| `GET /api/health` | 健康检查（免登录） |
| `POST /api/vision/count` | 羊群计数（multipart 上传图片，返回数量与标注图） |
| `POST /api/vision/count-video` | 视频计数（上传视频 ≤15s → 逐帧检测 + 跟踪 + 越线统计，返回标注帧序列与三条口径） |
| `POST /api/vision/coverage` | 草地覆盖度与绿度（未关联草场的独立分析） |
| `POST /api/vision/orthorectify` | 塔基斜拍图 → 四点标定单应变换正射校正（返回 H 矩阵、重投影误差与校正前后覆盖度对比） |
| `POST /api/vision/segment` | 草场图 → 自动划分管理分区（植被掩膜连通域聚类）+ 每区覆盖度 + 彩色叠加图 |
| `POST /api/vision/carrying-capacity` | 载畜量计算（NY/T 635） |
| `GET /api/pastures` | 草场清单（本草场覆盖度/告警等级/照片与分区数/载畜量建议） |
| `POST /api/pastures` | 新建草场（编号唯一） |
| `GET /api/pastures/{id}` | 草场详情：分区明细与编号照片的最新分析值 |
| `POST /api/pastures/{id}/images` | 按编号上传草场照片（编号即归属：草场-分区-序号）→ 覆盖度分析入库 |
| `POST /api/pastures/{id}/analyze` | 一键分析该草场全部照片（照片集合 → 草场状况） |
| `GET /api/pastures/{id}/trend` | 该草场按日聚合的覆盖度趋势；`forecast_days=7` 时附 Holt 线性趋势预测与置信带 |
| `DELETE /api/pastures/{id}` | 删除草场（照片与覆盖度记录一并移除） |
| `POST /api/agent/chat` | 自然语言 → 工具调用 → 决策 JSON |
| `POST /api/agent/stream` | 同上，SSE 逐步推送思考/工具调用/结果/决策事件（大屏思考流） |
| `GET /api/agent/history` | 历史对话列表（含思考事件，供 AI 助手界面回放） |
| `GET /api/agent/history/{id}` | 单条历史对话（大屏决策流点击跳转到对应回答） |
| `GET /api/dashboard/overview` | 概览 KPI：逐场草场状态（各草场独立）+ 等级取最严重 + 载畜量合计 + 羊数/温湿度 |
| `GET /api/dashboard/environment` `/trends` `/decision-flow` | 大屏数据（`/trends` 覆盖度**每个草场一条序列**，不做跨草场平均） |
| `POST /api/patrol/run` | 手动巡检一轮（**逐草场**检测 → 评估 → 需告警时落库并推送；返回每个草场的结果） |
| `GET /api/dashboard/alerts` | 巡检告警历史（含草场归属，最新在前；面板与轮询兜底共用） |
| `POST /api/dashboard/alerts/{id}/ack` | 确认告警（大屏「已知悉」） |
| `GET /api/dashboard/alerts/stream` | 告警实时推送（SSE，事件名 `alert`；含 15s 心跳） |

统一响应结构 `{code, data, message}`（code=0 成功）。

**鉴权**：除 `/api/health` 与 `/api/auth/login` 外，全部 `/api/*` 需要 `Authorization: Bearer <token>` 或 `?token=<token>`（SSE 用后者——EventSource 无法自定义请求头）。令牌为 HMAC-SHA256 签名的无状态令牌（纯标准库实现，无第三方依赖），退出登录为客户端清除令牌（无状态设计不做服务端吊销）。

## 演示脚本

服务启动后：

```bash
bash scripts/demo.sh
```

依次走通：健康检查 → 大屏/后台页面 → 概览 KPI → 趋势与环境 → 上传计数 → 覆盖度 → 载畜量 → Agent 对话 → 决策流。

## 测试

```bash
python -m pytest
```

全部测试使用 mock 与合成数据，不依赖网络、权重与 GPU，CPU/GPU 两种环境均可运行。

## 关键设计

- **登录鉴权与账号（真实校验）**：账号存于 `users` 表（首次启动用 `.env` 凭据播种管理员），口令只存 **PBKDF2-HMAC-SHA256 加盐哈希**（12 万轮，纯标准库，任何接口不回显）；登录签发标准库 HMAC-SHA256 无状态令牌（先验签后解析、常量时间比较、TTL 12h 可调）；**纯 ASGI 中间件**保护 `/api/*`（刻意不用 `BaseHTTPMiddleware`——不缓冲 SSE、`http.disconnect` 直传，告警订阅不泄漏、60MB 视频上传不受影响）；令牌双通道（Bearer 头 / `?token=` 查询参数供 EventSource）。**角色权限（简单模型）**：管理员可维护账号（`/api/users` 全部接口 403 拦截非管理员），其余页面登录即可访问；按角色的菜单/数据权限（RBAC）规划中。前端全局 fetch 包装只接管 `/api/*`、绝不消费响应体（Agent 思考流 `getReader()` 不受影响），401 统一回登录屏。
- **确定性优先**：告警等级由覆盖度阈值规则判定；载畜量 = 理论载畜量（NY/T 635）× 告警系数（正常 1.0 / 预警 0.6 / 危险禁牧 0）；**每个草场独立核定**（多草场时等级取各场中最严重、载畜量取各场之和）。理论值在覆盖度已知时按折算亩产自动计算（以 60% 覆盖度为满产参照线性折算 800kg/亩）。LLM 只补充建议文字，数字均有公式出处、可复现、可单测。
- **草场独立口径**：一个草场内部才做聚合（分区照片 → 本草场覆盖度），**草场之间彼此独立**——概览/趋势/决策/巡检均逐场返回，不存在跨草场的覆盖度平均值；跨草场只做状态汇总（最严重等级）与可加量求和（载畜量、面积）。
- **塔基图像几何校正**：塔基摄像头俯角大、缺少地理参考，斜拍图直接分析会低估覆盖度（透视压缩）。四点标定 → 单应矩阵 H → 正射校正（`cv2.getPerspectiveTransform`），前端 canvas 点选交互，返回重投影误差与校正前后覆盖度对比。灵感与思路来源：覃星力等《AI 驱动的塔基农情遥感监测新技术》（中科院空天院 CropWatch，2026）关键技术 1 的**轻量实现**（论文为云台参数 + 四元数自动配准；本项目为四点标定单应变换，报告与文档中如实标注差异）。
- **草场分区自动分割**：植被掩膜（ExG+HSV）→ 形态学开闭 → 连通域分析 → 空间聚类，把一张草场图自动划分为 K 个管理分区（A/B/C…）并给分区覆盖度与彩色叠加图。呼应上述论文关键技术 2（空间信息引导的 SAM 分割）的**轻量实现**（无地理参考下的纯图像连通域聚类，全流程确定性）。
- **覆盖度时序预测**：Holt 线性趋势外推（手写实现，无第三方依赖）+ 随步长加宽的置信带（1.96 × 残差标准差 × √步长），定位为趋势预警辅助；预测值按覆盖度物理边界截断到 [0,1]。
- **多模态融合草情评分**：覆盖度（60%，60% 为满产参照）+ 绿度 GLI（15%）+ 环境温湿度适宜度（25%）确定性加权，缺项自动归一化；等级与全站告警口径一致，评分用于展示与巡检文案（如「多模态草情评分 49/100」）。
- **Agent 思考流**：`POST /api/agent/stream` 以 SSE 实时推送 Orchestrator 的每一步（思考文字/工具调用/工具结果/最终决策），前端打字机滚动呈现；无 key 时规则降级过程同样可见。
- **Agent 与驾驶舱同口径**：草场类问题优先调用 `pasture_status`（草场状况，逐场返回）；命中单一草场用该场数据核定，未命中则**逐场列出**（等级取最严重、载畜量取各场之和），与草场分析视图、大屏使用同一份结果；参数试算（亩产/面积/放牧天数）才走 `carrying_capacity`；模型未调用工具时自动用规则补齐，保证决策数字全部来自工具。
- **主动巡检与实时预警（阶段 7）**：`逐场巡检 → 确定性判定 → 落库 → SSE 推送大屏`。**巡检单位是草场**：一轮巡检逐个检查每个草场（各自的编号照片聚合状况），冷却与升级判定按本场历史互不影响；手动巡检返回**每个草场的结果**（含未告警的草场），大屏面板顶部逐场展示。「升级必发、同级冷却抑制、降级发恢复通知」为纯函数规则（阈值复用全站唯一口径）；后台巡检任务默认关闭（`PATROL_ENABLED`）；升级时用专用 LLM 客户端（`max_retries=0` + 15s 超时）补一句建议，失败不影响告警；大屏以 SSE 实时弹出 + 5 秒轮询双通道，断线不丢告警，点击弹窗定位并高亮告警面板。
- **固定输入推理**：检测输入统一 letterbox 到 640×640 后再推理（张量形状恒定），规避 ROCm/MIOpen 对未编译形状长时间编译甚至卡死的问题，检测框自动映射回原图坐标。
- **视频计数（阶段 6）**：`detect() → ByteTrack 跟踪 → LineZone 越线` 三段解耦；三条口径分开返回（同帧最大并发 / 累计出现过 / 进·出·净），不编码 MP4 而是返回**标注帧序列**（JPEG base64）由前端 canvas 播放，画面与数字天然同步。跟踪用 supervision（scipy 匹配），刻意避开 ultralytics `model.track()` 的 lap 运行时静默安装（离线演示会直接 500）。
- **降级链路**：无 GLM key / 调用失败 → 关键字规则降级（瞬时错误自动指数退避重试）；无羊专用权重 → COCO 兜底；无 GPU → CPU 推理。
- **重资源惰性加载**：模型、数据库连接首次调用时初始化。

## 开源来源与许可

| 来源 | License | 用途 |
|---|---|---|
| [ICAERUS UC3 Livestock Monitoring](https://github.com/ICAERUS-EU/UC3_Livestock_Monitoring) | GPL-3.0 | 羊专用权重 `yolo11n_all.pt`、样例图（`data/samples/`）及测试图库（`data/samples/gallery/`：视频 `sheep_crossing.MP4` 抽帧 50 张 + 网格切分 24 张）。随包的 `yolo11n_all.pt` 仅清除了 checkpoint 内嵌的训练路径字符串（`scripts/strip_weight_paths.py`），权重张量未改动 |
| 本团队微调权重 `models/yolo11n_private.pt` | GPL-3.0（派生自上一行的 ICAERUS 权重） | 以上一行权重初始化，在私有数据集 Muyitong-Sheep 上微调 30 轮（2026-10-11，AMD RX 6800M / ROCm）得到的**修改版本**；499 个张量与 `runs/private_ft/weights/best.pt` 逐元素相同，仅清除了内嵌的本机绝对路径。按 GPL-3.0 分发 |
| [UAV footage of Merino sheep in South Africa](https://data.mendeley.com/datasets/nppwbkdf85/1)（Theart R., Schreve K.，Mendeley Data，2024-03-28） | CC BY 4.0（署名：Theart & Schreve, 2024） | **私有数据集 Muyitong-Sheep**（`data/private_dataset/`）由其处理而来：质量筛除→标注转换→密集切片→统一分辨率→确定性增强→分层划分，管线见 `scripts/make_private_dataset.py`； 羊群识别快速样例 10 张（`data/samples/sheep/`，编号 `P1-S-01`…`P3-S-03`）+ 草场分区编号照片 10 张（`data/samples/grassland/`，编号 `P1-A-01`…`P3-C-01`）：真实无人机影像（含 COCO 点标注真值）；分别由 `scripts/make_sheep_samples.py` / `scripts/make_grassland_samples.py` 确定性生成，原始数据包（>5GB）不入库 |
| [geezacoleman/OpenWeedLocator](https://github.com/geezacoleman/OpenWeedLocator) | MIT | `exg()` / `exg_standardised()` / `exg_standardised_hue()` 植被分割算法移植（`app/vision/grassland.py`，保留来源注释） |
| [ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | 推理框架（作库调用，未修改源码） |
| [apache/echarts](https://github.com/apache/echarts) | Apache-2.0 | 大屏图表库（CDN 引入：npmmirror 优先、jsdelivr 兜底） |
| [xiaofuqing13/echarts-bigscreen-skill](https://github.com/xiaofuqing13/echarts-bigscreen-skill) | MIT | 单文件大屏 + 1920×1080 缩放 + CDN fallback 规范参考 |
| 覃星力《AI 驱动的塔基农情遥感监测新技术》（中科院空天信息创新研究院 CropWatch 团队，2026） | 学术报告 | 塔基监测场景与「本地快速检测 + 云端深度分析」架构的**思路来源**；本项目实现其关键技术 1（几何校正）与关键技术 2（地块分割）的轻量简化版，差异在文档中如实标注 |
| Unsplash 摄影素材 | [Unsplash License](https://unsplash.com/license)（免费商用） | 界面背景/纹理：`bg-pasture.jpg`（photo-1533415648777-407b626eb0fa）、`sheep-flock.jpg`（photo-1484557985045-edf25e08da73），已本地化至 `static/assets/` |
| ICAERUS 真实无人机俯拍牧地（`data/samples/mauron_example1.JPG`，见上行 ICAERUS 来源） | GPL-3.0 | **塔基斜拍模拟样例** `static/assets/samples/tower_oblique.jpg`：由该真实俯拍牧地经已知单应变换确定性生成（`scripts/make_tower_sample.py`，演示资产）；**分区自动分割页「载入样例」**同样直接使用该真实俯拍图 |
| Pexels 视频素材 | [Pexels License](https://www.pexels.com/license/)（免费商用、无需署名） | 视频计数演示素材（`data/samples/video/`）：`aerial_flock.mp4`（Pexels 9906461 航拍羊群，12.35s）、`sheep_road.mp4`（Pexels 20039671 公路羊群，5.12s 竖屏） |
| 智谱 GLM（open.bigmodel.cn） | 服务条款 | LLM 推理服务（OpenAI 兼容 API） |
| NY/T 635《天然草地合理载畜量的计算》 | 行业标准 | 载畜量公式：载畜量 = 产量 × 利用率 ÷ (日食量 × 放牧天数)，1 牛 ≈ 5 羊单位 |
