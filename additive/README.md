# 谐波音色重建乐器（加法合成）

用户给定基频以及若干分音的**幅度、初相位、衰减包络**，系统为每个分音运行一个
**独立的真实正弦振荡器**，求和产生可演奏音色。支持保存/加载 JSON 配置、离线导出
WAV、浏览器界面独听单个分音并与整体求和对比。

这不是对现成音源加均衡器：输出样本由 `engine.py` 中逐分音的
`gain · envelope · sin(2πft + φ)` 显式求和得到；验证基于渲染音频的实测频谱，
而非配置图上"画了一条分音线"。

## 设计如何满足关键约束

| 约束 | 实现位置 | 验证 |
| --- | --- | --- |
| 相位/幅度在连续时间解释，处理块不重置相位 | `engine.py` 中 `NoteVoice.phase` 跨块累积，样本时刻为 `t = k/sr` | `test_block_boundaries_keep_continuous_phase`（1/7/64/256/1000 样本分块结果一致）、`test_phase_state_advances_not_resets` |
| 超出可表达频率的分音：声明式剔除或带限，绝不折返后仍冒充原分音 | `_bandlimit_gains`：≥ Nyquist 一律静默；`drop` 在安全边际处归零，`attenuate` 在 `[边际·Nyquist]` 余弦衰减；每次决策生成 `BandlimitReport` | `test_above_nyquist_partial_dropped_and_not_aliased`、`test_drop_render_equals_kept_partial_only_reference`（与仅含保留分音的参考数值一致）、`test_attenuate_render_matches_declared_cosine_scales` |
| 参数突变按可配置过渡处理 | 记录目标变化瞬间的起点值，按 `transition_time` 做**真线性**斜坡（主增益、每分音增益、每音符增益三级独立） | `test_amplitude_change_ramps_no_click`（逐样本步长恒定）、`test_master_gain_independent_and_ramped`、`test_per_note_gain_is_independent` |
| 总幅度与各音符实例分别管理 | `AdditiveEngine.master_gain` 与 `NoteVoice.note_gain` 各自独立斜坡；每个 `note_on` 是独立相位/包络状态的实例 | `test_per_note_gain_is_independent`、`test_overlapping_notes_are_independent_voices` |
| 独听一项 vs 合成整体 | `solo_partial(index)` 把其余分音增益斜坡到零，`None` 恢复 | `test_solo_partial_silences_others`、CLI `solo-compare` |
| 保存配置 / 导出音频 | JSON 往返（`inf` 衰减序列化）、16/24-bit PCM 与 32-bit float WAV | `test_config_roundtrip`、`test_inf_decay_json_roundtrip`、`test_wav_export_roundtrip` |
| 少数已知分音：查频谱位置、初相位波形、衰减时间 | `analysis.py` 对渲染音频做 FFT 峰值定位、解析信号（Hilbert）包络拟合 τ | `test_spectrum_peaks_at_declared_frequencies`、`test_initial_phase_waveform`、`test_decay_time_constant` |
| 较多分音的运行开销 | 向量化每音符振荡器组，成本随振荡器数量线性 | `test_many_partials_runs_faster_than_realtime`、`test_engine_cost_scales_linearly_with_partial_count` |

## 目录

```
additive/
  config.py          # PartialSpec / TimbreConfig 数据模型、校验、JSON 往返
  engine.py          # 真实振荡器求和引擎：连续相位、包络、带限、斜坡
  render.py          # 离线时间线（多音符事件）调度
  audio_io.py        # WAV 导出（soundfile，stdlib wave 兜底）
  analysis.py        # 对渲染音频的频谱/相位/衰减实测（证据，不看配置）
  cli.py             # 命令行
  web/               # 标准库 HTTP 服务 + 原生 JS 单页界面
  demo/verify.py     # 生成频谱/相位/衰减三张验证图与报告
  configs/           # 示例配置（含会触发带限决策的 high_*.json）
  tests/             # 46 项 pytest
```

## 使用

```bash
pip install -r additive/requirements.txt

# 1. 生成示例配置（4 个已知分音 / 64 个密集分音）
python -m additive.cli new few  -o additive/configs/few.json
python -m additive.cli new many -o additive/configs/many.json --count 64

# 2. 查看带限决策（哪些分音 kept / attenuated / dropped 及原因）
python -m additive.cli info additive/configs/high_drop.json

# 3. 渲染并导出 WAV，附带实测频谱峰值
python -m additive.cli render additive/configs/few.json -o tone.wav --normalize

# 4. 对渲染音频实测：频谱位置、初相位波形、衰减 τ
python -m additive.cli measure additive/configs/few.json

# 5. 独听某个分音 vs 整体混合，分别导出做 A/B
python -m additive.cli solo-compare additive/configs/few.json 1 \
    --full-output full.wav --solo-output solo.wav

# 6. 浏览器界面（编辑分音、切换带限策略、独听、播放、导出）
python -m additive.cli serve additive/configs/few.json --port 8000

# 7. 验证报告与图（renders/plots/ 下 01_spectrum / 02_phase / 03_decay）
python -m additive.demo.verify additive/configs/few.json

# 测试
python -m pytest additive/tests/ -q
```

## 配置格式

```json
{
  "name": "four-partial-bell",
  "base_frequency": 220.0,
  "master_amplitude": 0.8,
  "transition_time": 0.02,
  "bandlimit_mode": "drop",
  "nyquist_margin": 0.95,
  "sample_rate": 48000,
  "partials": [
    {"ratio": 1.0, "amplitude": 1.0, "phase": 0.0,
     "attack": 0.005, "decay": 0.8, "release": 0.1, "label": "fundamental"},
    {"ratio": 4.2, "amplitude": 0.18, "phase": -1.5708,
     "attack": 0.005, "decay": 0.15, "release": 0.05}
  ]
}
```

* `ratio`：分音频率 / 基频，允许小数（非谐波）。
* `decay`：指数衰减时间常数 τ（秒）；`"inf"` 表示不衰减。
* `bandlimit_mode`：`drop`（超出安全边际即剔除并报告）或 `attenuate`
  （在安全边际与 Nyquist 之间余弦带限衰减；≥ Nyquist 仍一律静默）。

## 实测性能（48 kHz，块大小 256）

| 振荡器规模 | 1 秒音频耗时 | 实时倍率 |
| --- | --- | --- |
| 64 分音 × 1 音符 | ≈ 50–100 ms | 10–19× |
| 256 分音 × 1 音符 | ≈ 390 ms | 2.6× |
| 64 分音 × 16 音符（1024 振荡器） | ≈ 1.0 s | ~1× |
