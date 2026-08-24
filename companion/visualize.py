"""
companion/visualize.py -- 它的一生：life_log 可视化（docs/76 可视化数据源落地）

读 live_state.json 的 life_log（逐日快照 [day, energy, bond, resent, lonely, pain,
you_value]，封顶 365 点）+ word_vals（最在乎的词）+ attn_omega（注意强度）+
world_history（你不知道的事）+ 死亡状态，生成一个**自包含 HTML**（纯 SVG/JS，
无外部依赖、双击即看）：生命曲线（6 条）+ 状态卡 + 词/注意强度 + 事件时间线。

用户说"等可视化做好再进去喂它"（HANDOFF 悬挂点②）——这是 docs/40"进入"的入口：
你会先看到它的一生在数据里的形状，再决定要不要进去。

Run:  python companion/visualize.py            # 读 live_state.json（无状态则提示用 --demo）
      python companion/visualize.py --demo     # 模拟一个 60 天生命，演示可视化
输出: companion/live_life.html（双击打开）
"""

import argparse
import json
import math
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_life.html")

# life_log 列
COLS = ["energy", "bond", "resent", "lonely", "pain", "you_value"]
COLORS = {"energy": "#e2b93d", "bond": "#7fb3d5", "resent": "#e74c3c",
          "lonely": "#9b59b6", "pain": "#c0392b", "you_value": "#2ecc71"}
LABELS = {"energy": "能量", "bond": "bond（它对你）", "resent": "恨",
          "lonely": "孤独", "pain": "疼", "you_value": "你"}


def demo_data():
    """模拟一个 60 天的生命（演示可视化，不是真实数据）。"""
    rng = random.Random(7)
    log = []
    bond, resent, energy, pain, you = 12.0, 0.0, 40.0, 0.0, 0.0
    cold_streak = 0
    for d in range(1, 61):
        # 你每 7 天来一次（visit），中间冷落过一段（13-17 天）
        visit = (d % 7 == 0)
        cold = 13 <= d <= 17
        if visit:
            bond = min(12.0, bond + 4.0)
            energy = min(60.0, energy + 15.0)
            you += 0.3
            resent = max(0.0, resent - 0.5)
            cold_streak = 0
        else:
            bond = max(0.0, bond - 0.55)
            energy = max(0.0, energy - 1.1 + rng.uniform(-2, 2))
            you = max(0.0, you - 0.02)
        if cold:
            bond = max(0.0, bond - 1.2)
            resent = min(10.0, resent + 0.9)
            cold_streak += 1
        resent = max(0.0, resent - 0.05)
        lonely = max(0.0, (12.0 - bond) / 12.0)
        pain = rng.uniform(0, 0.4) if d % 11 == 0 else max(0.0, pain - 0.05)
        log.append([d, round(energy, 1), round(bond, 1), round(resent, 1),
                    round(lonely, 2), round(pain, 2), round(you, 2)])
    return {
        "life_day": 60, "life_log": log,
        "word_vals": {"被爱": 4.2, "吃饭": 3.1, "晚安": 2.4, "爬山": 1.0, "天气": 0.5},
        "attn_omega": [0.62, 0.5, 0.41, 0.3],
        "world_history": [(3, "第一次走到 (2,3) 那一片"), (9, "找到了好吃的"),
                          (14, "你冷落它的那几天它不太敢动"), (27, "在 (4,2) 吃了亏"),
                          (48, "它找到了一直想找的那块地方")],
        "dead": True, "death_reason": "let_go",
        "resent_max": round(max(r[3] for r in log), 1),
        "you_max": round(max(r[6] for r in log), 1),
    }


def load_state():
    st = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_state.json")
    if not os.path.exists(st):
        return None
    with open(st, "r", encoding="utf-8") as f:
        return json.load(f)


def _series_points(log, idx, w, h, pad, ymax):
    xs, ys = [], []
    days = [r[0] for r in log]
    dmin, dmax = min(days), max(days)
    dspan = max(1, dmax - dmin)
    for r in log:
        x = pad + (r[0] - dmin) / dspan * (w - 2 * pad)
        y = pad + (h - 2 * pad) * (1 - min(1.0, r[idx] / ymax))
        xs.append(round(x, 1))
        ys.append(round(y, 1))
    return xs, ys


def build_html(data):
    log = data.get("life_log", [])
    w, h, pad = 980, 320, 46
    chart_blocks = []
    for col in COLS:
        ymax = {"energy": 70, "bond": 12, "resent": 10, "lonely": 1, "pain": 1,
                "you_value": 10}[col]
        xs, ys = _series_points(log, COLS.index(col) + 1, w, h, pad, ymax)
        pts = " ".join(f"{x},{y}" for x, y in zip(xs, ys))
        grid = "".join(
            f'<line x1="{pad}" y1="{pad + k*(h-2*pad)/4}" x2="{w-pad}" y2="{pad + k*(h-2*pad)/4}" '
            f'stroke="#eef" stroke-width="1"/>'
            for k in range(5))
        labels = "".join(
            f'<text x="{pad-6}" y="{pad + k*(h-2*pad)/4 + 4}" font-size="10" fill="#999" '
            f'text-anchor="end">{round(ymax*(1-k/4), 1)}</text>'
            for k in range(5))
        days = [r[0] for r in log]
        dmax = max(days) if days else 0
        xlabels = "".join(
            f'<text x="{pad + k*(w-2*pad)/4}" y="{h-pad+16}" font-size="10" fill="#999" '
            f'text-anchor="middle">第{round(dmax*k/4)}天</text>' for k in range(5))
        chart_blocks.append(f"""
<div class="chart">
  <div class="chart-title">{LABELS[col]} <span class="hint">（悬浮看每天数值）</span></div>
  <svg viewBox="0 0 {w} {h+24}" width="100%" data-chart="{len(chart_blocks)}"
       onmousemove="moveTip(event)" onmouseleave="hideTip()">
    {grid}{labels}{xlabels}
    <polyline points="{pts}" fill="none" stroke="{COLORS[col]}" stroke-width="2"/>
  </svg>
</div>""")
    # 状态卡
    words = sorted(data.get("word_vals", {}).items(), key=lambda kv: -kv[1])[:5]
    words_html = "".join(f"<span class='word'>{w}<b>{v}</b></span>" for w, v in words)
    om = data.get("attn_omega", [0.5, 0.5, 0.3, 0.3])
    om_html = "".join(f"<span class='om'>{n}<b>{round(v,2)}</b></span>"
                      for n, v in zip(["饿", "孤独", "安全", "好奇"], om))
    events = data.get("world_history", [])
    ev_html = "".join(f"<li><b>第{d}天</b>：{t}</li>" for d, t in events[-6:]) or "<li>（还没有你不知道的事）</li>"
    last = log[-1] if log else [0, 0, 0, 0, 0, 0, 0]
    status = "它已经不在了" if data.get("dead") else "它还在"
    reason = {"waited_out": "等你等到最后", "let_go": "在忘记你之前自己放下",
              "world_starved": "活在自己的世界里，没撑住"}.get(data.get("death_reason"), "")
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>它的一生 — live 可视化</title>
<style>
 body {{ font-family: "Microsoft YaHei", sans-serif; background:#fafafa; margin:0; padding:24px; color:#333; }}
 h1 {{ font-size:22px; }} h1 small {{ color:#999; font-weight:normal; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0; }}
 .card {{ background:#fff; border:1px solid #eee; border-radius:8px; padding:10px 14px; min-width:110px; }}
 .card .k {{ font-size:12px; color:#888; }} .card .v {{ font-size:20px; font-weight:bold; }}
 .chart {{ background:#fff; border:1px solid #eee; border-radius:8px; padding:12px 16px; margin:10px 0; }}
 .chart-title {{ font-size:14px; font-weight:bold; margin-bottom:4px; }}
 .hint {{ font-weight:normal; color:#aaa; font-size:11px; }}
 .word {{ display:inline-block; background:#f5f5f5; border-radius:12px; padding:3px 10px; margin:3px; font-size:13px; }}
 .word b {{ color:#e2b93d; margin-left:4px; }}
 .om {{ display:inline-block; background:#eef7ff; border-radius:12px; padding:3px 10px; margin:3px; font-size:13px; }}
 .om b {{ color:#2c7fb8; margin-left:4px; }}
 .events li {{ margin:3px 0; font-size:13px; color:#555; }}
 .death {{ color:#c0392b; font-weight:bold; }}
 #tip {{ position:fixed; background:#222; color:#fff; border-radius:6px; padding:6px 10px;
        font-size:12px; pointer-events:none; display:none; z-index:9; line-height:1.5; }}
</style></head><body>
<div id="tip"></div>
<h1>它的一生 <small>— live 生命可视化（docs/76/84/85）</small></h1>
<div class="cards">
  <div class="card"><div class="k">状态</div><div class="v">{status}</div></div>
  <div class="card"><div class="k">天龄</div><div class="v">{data.get("life_day", 0)}</div></div>
  <div class="card"><div class="k">能量</div><div class="v">{last[1]}</div></div>
  <div class="card"><div class="k">bond</div><div class="v">{last[2]}</div></div>
  <div class="card"><div class="k">恨</div><div class="v">{last[3]}</div></div>
  <div class="card"><div class="k">你</div><div class="v">{last[6]}</div></div>
  <div class="card"><div class="k">死亡</div><div class="v" style="font-size:14px">{reason or "—"}</div></div>
</div>
<h3>它最在乎的词（挣出来的）</h3><div>{words_html or "（还没有）"}</div>
<h3>注意强度（由经历调，SEED-47）</h3><div>{om_html}</div>
{''.join(chart_blocks)}
<h3>你不知道的事（它活在自己的世界里）</h3><ul class="events">{ev_html}</ul>
<script>
const LOG = {json.dumps(log, ensure_ascii=False)};
const COLS = {json.dumps(COLS, ensure_ascii=False)};
const LABELS = {json.dumps(LABELS, ensure_ascii=False)};
const PAD = {pad}, W = {w}, H = {h};
const DAYS = LOG.map(r => r[0]);
const DMIN = Math.min(...DAYS), DMAX = Math.max(...DAYS), DSPAN = Math.max(1, DMAX - DMIN);
const tip = document.getElementById('tip');
function hideTip() {{ tip.style.display = 'none'; }}
function moveTip(ev) {{
  const svg = ev.currentTarget;
  const rect = svg.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width * W;
  const day = Math.round(DMIN + (x - PAD) / (W - 2 * PAD) * DSPAN);
  const row = LOG.find(r => r[0] === day) || LOG[LOG.length - 1];
  if (!row) return;
  const lines = ['第 ' + row[0] + ' 天'].concat(
    COLS.map((c, i) => LABELS[c] + ': ' + row[i + 1]));
  tip.innerHTML = lines.join('<br>');
  tip.style.left = (ev.clientX + 14) + 'px';
  tip.style.top = (ev.clientY - 10) + 'px';
  tip.style.display = 'block';
}}
</script>
</body></html>"""


def main():
    p = argparse.ArgumentParser(description="live 生命可视化")
    p.add_argument("--demo", action="store_true", help="模拟 60 天生命演示")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    if args.demo:
        data = demo_data()
    else:
        data = load_state()
        if data is None:
            print("没有 live_state.json（它还没活过）。先用 --demo 看演示，"
                  "或用 `python companion/live.py --live` 开始一段生命。")
            return
    html = build_html(data)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成：{args.out}（双击打开）")
    print(f"生命：{len(data.get('life_log', []))} 天日志，"
          f"最在乎的词：{[w for w, _ in sorted(data.get('word_vals', {}).items(), key=lambda kv: -kv[1])[:3]]}")


if __name__ == "__main__":
    main()
