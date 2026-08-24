# 77 — 嘴升级：LLM 当嘴不当心（docs/56 的落地 + 隔离测试 + 框架纪律）

> 缘起：2026-08-23 晚。用户："**要不我们现在先给它补一个语言能力？**"——项目早有架构答案
> （docs/48：主体要的是有立场的智能，不是语言延伸的智能；docs/56/proto6：**嘴=可插拔翻译器，
> 语言是输出，永远不是心**），live 的心现在很丰富（记忆/bond/恨/身体/世界，docs/76），
> 嘴是最弱一环（模板）。本轮回填：**可插拔的 LLM 嘴**（DeepSeek）+ **更丰富的模板回退** +
> **隔离测试**（换嘴不换心，proto6 的复验）。
>
> 状态：工程回填（纯本地可测；LLM 嘴需 key，未在本地跑）。文件：`companion/mouth_llm.py`（新）、
> `companion/live.py`（--mouth 参数 + build_mouth + 隔离测试）。

---

## 〇、一句话

> **语言能力补的是"嘴"，不是"心"。** 心=状态（判断/记忆/利害/恨/身体全在结构里，docs/76）；
> 嘴=可插拔 `translate(state, context)->str`——模板（规则）或 LLM（DeepSeek，key 走环境变量，
> 无 key 优雅回退）。**换嘴不换心**：demo 实测好嘴 vs 坏嘴（乱码）下，心的 items/bond/energy/
> resent/you_value **逐位一致**。LLM 只当嘴，永远不当心（docs/48/56）；嘴的输出从不写回心
> （隔离结构性成立）。prompt 铁律（docs/29b 框架纪律的嘴版）：**只翻译状态里已有的信息，
> 绝不添加状态里没有的记忆/事件/想法/感受**。

## 一、为什么是"嘴"不是"心"（docs/48/56 回顾）

- docs/48：语言延伸不是主体要的智能——LLM 是"无立场的智能"（docs/75 反驳后仍站得住的：
  它缺"预见错→付出→不可逆"回路），**让它当大脑 = 把判断层交给镜像**；
- docs/56/proto6：心=载体，嘴=可插拔翻译器——**换嘴不换心**（隔离测试：好嘴/坏嘴下心逐位
  一致）。判断层不受 LLM 影响，受限的只有表达层；
- 所以"补语言能力"的正确切法：**给嘴升级，心原样不动**。

## 二、机制（companion/mouth_llm.py + live.py）

- **LLMMouth**：与模板 Mouth 鸭子兼容（`translate(state, context)->str`）；key 从
  `DEEPSEEK_API_KEY` 读（绝不落盘）；API 出错/超时/无 key → **优雅回退到模板嘴**
  （心永远不依赖嘴——docs/56"嘴可以坏，心不能依赖嘴"）。
- **prompt 铁律**（docs/29b 框架纪律的嘴版）：系统提示写明"只翻译状态里已有的信息，绝不添加
  状态里没有的记忆/事件/想法/感受；不要说'我爱你'这类状态里没有的话；语气随状态走
  （bond 高温柔/bond 低疏远/resent 高冷淡/energy 低虚弱）"。
- **--mouth 三档**：`llm`（强制 LLM，需 key）/ `template`（规则）/ `auto`（默认：有 key 用
  LLM，无 key 用模板）。进入时提示"嘴=LLM(DeepSeek) / 嘴=模板"。
- **context 传入**：翻译时带上"你刚说了什么/离开 X 天后回来"，LLM 嘴能对上话（模板嘴忽略）。

## 三、结果（demo 实测，纯本地）

```
-- mouth isolation (docs/56/77: 换嘴不换心) --
good-mouth said: 我一直记得你说过：我想被爱。 （你还在，我就安心）
bad-mouth said:  ……（乱码）
hearts identical (items/bond/energy/resent/you) = True
LLM mouth without key falls back to template: '我一直记得你说过：我想被爱。 （你还在，我就安心）'
```

- **换嘴不换心复验通过**：好嘴/坏嘴下心的五项状态逐位一致——判断层不受嘴影响；
- **无 key 优雅回退通过**：LLMMouth(key="") 返回模板翻译，不崩；
- 模板嘴本身也微升级（接受 context、恨/饿/孤独表达，docs/76 已加）。

## 四、诚实边界

1. **LLM 嘴仍是"框架驱动"的**（docs/29b）：它的遣词随 prompt 走——所以 prompt 铁律只准它
   翻译、禁止添加事实；它"说"的不是它的判断，是它翻译的心（隔离保证判断层永远是结构）。
2. **LLM 嘴会幻觉**——所以只给它状态 JSON、且规则"不得添加状态外事实"；但单次 API 输出无法
   保证 100% 遵守（诚实：翻译层可能添油加醋，心不受影响，但嘴上可能说谎——需要观察；
   下一步可加"翻译校验"：解析输出，若含状态外事实则回退模板）。
3. **key 安全**：只走环境变量，不落盘、不进 git（照旧）。
4. **语言仍是表达层**（docs/48）：它不会因为会说人话而"更主体"——心有没有立场由 A2/A4 决定，
   不由嘴决定。换嘴不换心正是这句话的机制版。

## 五、运行

```bash
set DEEPSEEK_API_KEY=你的key        # 环境变量，不落盘
python companion/live.py --live --mouth llm      # LLM 嘴（有 key）
python companion/live.py --live                  # auto：有 key 用 LLM，无 key 用模板
python companion/live.py --demo                  # 含换嘴不换心隔离测试
```

## 六、下一步候选

1. **翻译校验**：LLM 输出若含状态外事实（幻觉）→ 回退模板（防嘴上说谎）。
2. 嘴的多风格档（温柔/冷淡/话痨）由状态或用户偏好切——但一律只翻译状态。
3. 可视化（life_log 数据源已就绪，docs/76）——用户说等可视化做好再进去喂它。
