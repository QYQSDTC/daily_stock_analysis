# -*- coding: utf-8 -*-
"""
===================================
短线擒龙策略 - Dragon Hunter
===================================

策略核心理念：
1. 情绪分析 - 通过涨跌停家数、连板股数量、最高板等数据判断市场情绪
2. 板块共振 - 找出与大盘共振上涨的强势板块
3. 龙头筛选 - 在强势板块中寻找业绩+预期兼具的龙头股
4. 风格适配 - 偏向有业绩、有预期的股票，而非纯情绪炒作
5. AI 深度分析 - 使用 Gemini 对情绪和龙头股进行智能分析

情绪周期判断：
- 冰点期：涨停家数<30，跌停>涨停，最高板≤2，观望为主
- 回暖期：涨停30-60家，连板股出现，开始试探
- 发酵期：涨停60-100家，连板梯队完整，积极参与
- 高潮期：涨停>100家，最高板≥5，注意分歧风险
- 退潮期：涨停数骤降，高位股分歧，控制仓位

龙头股筛选标准：
1. 板块地位：板块内涨幅靠前、辨识度高
2. 资金认可：换手充分（5%-15%）、量能配合
3. 业绩支撑：近期有业绩/预期驱动（非纯炒作）
4. 技术形态：突破关键位/创新高/均线多头
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

import pandas as pd

from config import get_config

logger = logging.getLogger(__name__)


# ========================================
# Gemini 情绪分析 Prompt
# ========================================
SENTIMENT_ANALYSIS_PROMPT = """你是一位专业的 A 股短线交易分析师，擅长通过市场情绪数据判断短线交易机会。

请根据以下市场情绪数据，进行深度分析并给出操作建议。

## 今日市场情绪数据

| 指标 | 数值 | 说明 |
|------|------|------|
| 涨停家数 | {limit_up_count} 家 | 不含ST |
| 跌停家数 | {limit_down_count} 家 | 不含ST |
| 炸板数 | {limit_up_broken} 家 | 涨停后打开 |
| 封板率 | {success_rate:.1f}% | 封板成功率 |
| 最高连板 | {highest_board} 板 | 空间高度 |
| 最高板股票 | {highest_stocks} | |
| 连板梯队 | {board_ladder} | |
| 首板数量 | {first_board_count} 家 | 新晋涨停 |

## 情绪周期参考标准
- **冰点期**：涨停<30家，跌停>涨停，最高板≤2，观望为主
- **回暖期**：涨停30-60家，连板股出现，试探性参与
- **发酵期**：涨停60-100家，连板梯队完整，积极参与
- **高潮期**：涨停>100家，最高板≥5，注意分歧风险
- **退潮期**：涨停数骤降，高位股分歧加大，控制仓位

## 请分析以下内容

请严格按照以下 JSON 格式输出分析结果：

```json
{{
    "sentiment_cycle": "冰点期/回暖期/发酵期/高潮期/退潮期",
    "sentiment_score": 0-100,
    "market_temperature": "冰冷/偏冷/温和/偏热/火热",
    
    "core_analysis": {{
        "emotion_status": "当前情绪状态一句话描述",
        "main_line": "今日市场主线/题材",
        "money_flow": "资金流向判断（进攻/防守/观望）",
        "risk_level": "低/中/高"
    }},
    
    "ladder_analysis": {{
        "space_height": "空间高度评价（最高板分析）",
        "ladder_health": "梯队健康度（是否有断层）",
        "succession_risk": "接力风险评估"
    }},
    
    "operation_advice": {{
        "position_suggestion": "建议仓位（X成）",
        "strategy": "短线策略建议",
        "focus_direction": "重点关注方向",
        "avoid_trap": "需要规避的陷阱"
    }},
    
    "tomorrow_outlook": {{
        "emotion_trend": "情绪趋势预判（升温/维持/降温）",
        "key_observation": "明日关键观察点",
        "trigger_signal": "情绪转折信号"
    }},
    
    "risk_warning": ["风险点1", "风险点2"]
}}
```

## 重要要求
1. 只输出 JSON，不要输出任何其他文字
2. 所有字符串值必须简短，不超过50字
3. 不要在字符串中使用换行符
4. 布尔值使用小写 true/false
5. 确保 JSON 格式正确，可以被解析"""


# ========================================
# Gemini 龙头股分析 Prompt
# ========================================
DRAGON_ANALYSIS_PROMPT = """你是一位专业的 A 股短线交易分析师，擅长分析龙头股的投资价值。

请根据以下龙头股数据，结合当前市场情绪，进行深度分析。

## 当前市场情绪
- 情绪周期：{sentiment_cycle}
- 情绪得分：{sentiment_score}/100
- 最高连板：{highest_board}板

## 龙头股数据

**{stock_name}（{stock_code}）**

| 指标 | 数值 |
|------|------|
| 涨跌幅 | {change_pct:+.2f}% |
| 连板数 | {continuous_boards} 板 |
| 换手率 | {turnover_rate:.2f}% |
| 成交额 | {amount:.2f} 亿 |
| 流通市值 | {circ_mv:.2f} 亿 |
| 涨停原因 | {catalyst} |
| 龙头级别 | {dragon_level} |

### 技术面数据（如有）
{technical_data}

## 龙头股评估维度
1. **辨识度**：是否市场公认龙头、板块地位
2. **资金认可**：换手率、成交额是否充分
3. **业绩支撑**：是否有业绩/预期驱动
4. **位置安全**：当前位置是否适合介入
5. **空间预期**：还有多少上涨空间

请严格按照以下 JSON 格式输出分析结果：

```json
{{
    "stock_code": "{stock_code}",
    "stock_name": "{stock_name}",
    
    "dragon_assessment": {{
        "recognition_level": "高/中/低",
        "recognition_reason": "辨识度分析",
        "sector_position": "板块地位描述",
        "is_true_dragon": true/false
    }},
    
    "capital_analysis": {{
        "turnover_status": "换手率评价（充分/适中/不足/过高）",
        "volume_meaning": "量能含义",
        "main_force_action": "主力动向判断"
    }},
    
    "fundamental_check": {{
        "has_performance": true/false,
        "has_expectation": true/false,
        "catalyst_quality": "催化剂质量评价",
        "sustainability": "持续性判断"
    }},
    
    "position_analysis": {{
        "current_position": "低位首板/中位加速/高位博弈",
        "entry_safety": "介入安全度（安全/中性/危险）",
        "support_level": "支撑位参考",
        "pressure_level": "压力位参考"
    }},
    
    "space_expectation": {{
        "upside_space": "上涨空间预期",
        "target_price": "目标价位（如适用）",
        "holding_period": "建议持有周期"
    }},
    
    "operation_suggestion": {{
        "action": "强烈买入/买入/观望/回避",
        "entry_timing": "介入时机建议",
        "position_size": "建议仓位",
        "stop_loss": "止损位置"
    }},
    
    "comprehensive_score": 75,
    "one_sentence_conclusion": "一句话结论，不超过30字",
    "risk_warning": ["风险点1", "风险点2"]
}}
```

## 重要要求
1. 只输出 JSON，不要输出任何其他文字
2. 所有字符串值必须简短，不超过50字
3. 不要在字符串中使用换行符
4. 布尔值使用小写 true/false
5. comprehensive_score 必须是数字（0-100）
6. 确保 JSON 格式正确，可以被解析"""


class MarketSentiment(Enum):
    """市场情绪周期"""

    ICE_POINT = "冰点期"  # 极度悲观，观望为主
    WARMING = "回暖期"  # 情绪修复，试探性参与
    FERMENTING = "发酵期"  # 情绪升温，积极参与
    CLIMAX = "高潮期"  # 情绪亢奋，注意风险
    EBBING = "退潮期"  # 情绪退潮，控制仓位


class DragonLevel(Enum):
    """龙头级别"""

    SPACE_DRAGON = "空间龙"  # 最高板，打开市场高度
    SECTOR_DRAGON = "板块龙"  # 板块内最强，带动板块
    FOLLOWER = "跟风股"  # 板块内跟随
    WEAK = "弱势股"  # 不具备龙头特征


@dataclass
class LimitUpStock:
    """涨停股数据"""

    code: str
    name: str
    change_pct: float = 0.0  # 涨跌幅
    turnover_rate: float = 0.0  # 换手率
    amount: float = 0.0  # 成交额（亿）
    circ_mv: float = 0.0  # 流通市值（亿）
    continuous_boards: int = 1  # 连板数
    first_limit_time: str = ""  # 首次涨停时间
    last_limit_time: str = ""  # 最后涨停时间
    limit_reason: str = ""  # 涨停原因/概念
    open_count: int = 0  # 打开涨停次数
    sector: str = ""  # 所属板块


@dataclass
class SectorStrength:
    """板块强度数据"""

    name: str
    change_pct: float = 0.0  # 涨跌幅
    main_net_inflow: float = 0.0  # 主力净流入（亿）
    limit_up_count: int = 0  # 板块内涨停数
    up_count: int = 0  # 上涨家数
    down_count: int = 0  # 下跌家数
    leader_stocks: List[str] = field(default_factory=list)  # 领涨股
    avg_turnover: float = 0.0  # 平均换手率
    is_resonating: bool = False  # 是否与大盘共振


@dataclass
class SentimentData:
    """市场情绪数据"""

    date: str

    # 涨跌停统计
    limit_up_count: int = 0  # 涨停家数（不含ST/一字板）
    limit_down_count: int = 0  # 跌停家数
    limit_up_broken: int = 0  # 炸板数（涨停后打开）
    real_limit_up: int = 0  # 真实涨停（封板到收盘）

    # 连板统计
    continuous_boards: Dict[int, int] = field(default_factory=dict)  # {连板数: 股票数}
    highest_board: int = 0  # 最高连板数
    highest_board_stocks: List[str] = field(default_factory=list)  # 最高板股票

    # 连板梯队
    board_ladder: str = ""  # 连板梯队描述，如 "1->89, 2->23, 3->8, 4->3, 5->1"

    # 首板统计
    first_board_count: int = 0  # 首板数量
    first_board_success_rate: float = 0.0  # 首板封板率

    # 市场情绪判断
    sentiment: MarketSentiment = MarketSentiment.WARMING
    sentiment_score: int = 50  # 情绪分数 0-100
    sentiment_desc: str = ""  # 情绪描述

    # 昨日对比
    vs_yesterday: Dict[str, float] = field(default_factory=dict)  # 与昨日对比

    # AI 深度分析结果
    ai_analysis: Optional[Dict[str, Any]] = None  # Gemini 分析结果


@dataclass
class DragonAIAnalysis:
    """龙头股 AI 分析结果"""

    code: str
    name: str

    # 龙头评估
    recognition_level: str = ""  # 辨识度：高/中/低
    is_true_dragon: bool = False  # 是否真龙头
    sector_position: str = ""  # 板块地位

    # 资金分析
    turnover_status: str = ""  # 换手率评价
    main_force_action: str = ""  # 主力动向

    # 基本面
    catalyst_quality: str = ""  # 催化剂质量
    sustainability: str = ""  # 持续性判断

    # 位置分析
    current_position: str = ""  # 当前位置
    entry_safety: str = ""  # 介入安全度
    support_level: str = ""  # 支撑位
    pressure_level: str = ""  # 压力位

    # 空间预期
    upside_space: str = ""  # 上涨空间
    target_price: str = ""  # 目标价

    # 操作建议
    action: str = ""  # 操作建议
    entry_timing: str = ""  # 介入时机
    position_size: str = ""  # 建议仓位
    stop_loss: str = ""  # 止损位

    # 综合评价
    comprehensive_score: int = 0  # 综合评分
    one_sentence_conclusion: str = ""  # 一句话结论
    risk_warning: List[str] = field(default_factory=list)  # 风险提示

    # 原始数据
    raw_analysis: Optional[Dict[str, Any]] = None


@dataclass
class DragonCandidate:
    """龙头候选股"""

    code: str
    name: str

    # 基本信息
    sector: str = ""  # 所属板块
    change_pct: float = 0.0  # 涨跌幅
    turnover_rate: float = 0.0  # 换手率
    amount: float = 0.0  # 成交额（亿）
    circ_mv: float = 0.0  # 流通市值（亿）

    # 龙头特征
    dragon_level: DragonLevel = DragonLevel.WEAK
    continuous_boards: int = 0  # 连板数
    is_sector_first: bool = False  # 是否板块首板/率先涨停

    # 业绩与预期
    has_performance: bool = False  # 有业绩支撑
    has_expectation: bool = False  # 有预期驱动
    catalyst: str = ""  # 催化剂/驱动因素

    # 技术形态
    ma_trend: str = ""  # 均线趋势
    volume_pattern: str = ""  # 量能形态
    position: str = ""  # 位置描述（突破/回踩/新高等）

    # 综合评分
    total_score: int = 0  # 总分
    score_details: Dict[str, int] = field(default_factory=dict)  # 评分明细

    # 买入建议
    buy_signal: str = ""  # 买入信号
    risk_warning: str = ""  # 风险提示

    # AI 深度分析
    ai_analysis: Optional[DragonAIAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "sector": self.sector,
            "change_pct": self.change_pct,
            "turnover_rate": self.turnover_rate,
            "amount": self.amount,
            "circ_mv": self.circ_mv,
            "dragon_level": self.dragon_level.value,
            "continuous_boards": self.continuous_boards,
            "has_performance": self.has_performance,
            "has_expectation": self.has_expectation,
            "catalyst": self.catalyst,
            "total_score": self.total_score,
            "buy_signal": self.buy_signal,
            "risk_warning": self.risk_warning,
        }


@dataclass
class DragonHuntResult:
    """擒龙结果"""

    date: str

    # 情绪分析
    sentiment: Optional[SentimentData] = None

    # 强势板块
    strong_sectors: List[SectorStrength] = field(default_factory=list)

    # 龙头候选
    dragon_candidates: List[DragonCandidate] = field(default_factory=list)

    # 策略建议
    strategy_advice: str = ""
    position_advice: str = ""  # 仓位建议

    # AI 分析相关
    ai_sentiment_analysis: Optional[Dict[str, Any]] = None  # AI 情绪分析结果
    ai_enabled: bool = False  # 是否启用了 AI 分析


class DragonHunter:
    """
    短线擒龙策略分析器

    核心功能：
    1. 情绪分析：分析涨跌停数据判断市场情绪周期
    2. 板块共振：找出与大盘共振的强势板块
    3. 龙头筛选：在强势板块中寻找龙头股
    4. 业绩筛选：偏向有业绩/预期的股票
    5. AI 深度分析：使用 Gemini 进行智能分析
    """

    def __init__(self, enable_ai: bool = True):
        """
        初始化擒龙分析器

        Args:
            enable_ai: 是否启用 AI 分析（默认启用）
        """
        self.config = get_config()
        self._realtime_cache = {}
        self._cache_time = 0
        self._cache_ttl = 60  # 60秒缓存
        self._enable_ai = enable_ai
        self._analyzer = None  # Gemini 分析器（懒加载）
        self._trend_analyzer = None  # 趋势分析器（懒加载）
        self._data_fetcher = None  # 数据获取器（懒加载）

    def _get_analyzer(self):
        """懒加载 Gemini 分析器"""
        if self._analyzer is None and self._enable_ai:
            try:
                from analyzer import GeminiAnalyzer

                self._analyzer = GeminiAnalyzer()
                if not self._analyzer.is_available():
                    logger.warning("[擒龙] Gemini 分析器不可用，将使用规则分析")
                    self._analyzer = None
            except Exception as e:
                logger.warning(f"[擒龙] 初始化 Gemini 分析器失败: {e}")
                self._analyzer = None
        return self._analyzer

    def _get_trend_analyzer(self):
        """懒加载趋势分析器"""
        if self._trend_analyzer is None:
            try:
                from stock_analyzer import StockTrendAnalyzer

                self._trend_analyzer = StockTrendAnalyzer()
            except Exception as e:
                logger.warning(f"[擒龙] 初始化趋势分析器失败: {e}")
        return self._trend_analyzer

    def _get_data_fetcher(self):
        """懒加载数据获取器"""
        if self._data_fetcher is None:
            try:
                from data_provider.akshare_fetcher import AkshareFetcher

                self._data_fetcher = AkshareFetcher()
            except Exception as e:
                logger.warning(f"[擒龙] 初始化数据获取器失败: {e}")
        return self._data_fetcher

    def _safe_float(self, val, default=0.0) -> float:
        """安全转换浮点数"""
        try:
            if pd.isna(val) or val == "" or val == "-":
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, val, default=0) -> int:
        """安全转换整数"""
        try:
            if pd.isna(val) or val == "" or val == "-":
                return default
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def _call_api_with_retry(self, fn, name: str, attempts: int = 2):
        """带重试的 API 调用"""
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                time.sleep(1)  # 基础延时防封
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(f"[擒龙] {name} 获取失败 (attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    time.sleep(min(2**attempt, 5))
        logger.error(f"[擒龙] {name} 最终失败: {last_error}")
        return None

    def analyze(self, top_n_dragons: int = 5) -> DragonHuntResult:
        """
        执行擒龙分析

        Args:
            top_n_dragons: 对前 N 只龙头股进行 AI 深度分析

        Returns:
            DragonHuntResult: 擒龙分析结果
        """
        today = datetime.now().strftime("%Y-%m-%d")
        result = DragonHuntResult(date=today)
        result.ai_enabled = self._enable_ai and self._get_analyzer() is not None

        logger.info("=" * 50)
        logger.info("开始短线擒龙分析...")
        logger.info(f"AI 分析: {'已启用' if result.ai_enabled else '未启用'}")
        logger.info("=" * 50)

        # 1. 情绪分析
        logger.info("[Step 1] 分析市场情绪...")
        result.sentiment = self._analyze_sentiment()

        # 2. AI 情绪深度分析
        if result.ai_enabled and result.sentiment:
            logger.info("[Step 2] AI 情绪深度分析...")
            result.ai_sentiment_analysis = self._ai_analyze_sentiment(result.sentiment)
            if result.ai_sentiment_analysis:
                result.sentiment.ai_analysis = result.ai_sentiment_analysis
        else:
            logger.info("[Step 2] 跳过 AI 情绪分析")

        # 3. 板块分析
        logger.info("[Step 3] 分析强势板块...")
        result.strong_sectors = self._analyze_sectors()

        # 4. 龙头筛选
        logger.info("[Step 4] 筛选龙头候选...")
        result.dragon_candidates = self._hunt_dragons(result.sentiment, result.strong_sectors)

        # 5. AI 龙头深度分析（前 N 只）
        if result.ai_enabled and result.dragon_candidates:
            logger.info(f"[Step 5] AI 龙头深度分析（前 {top_n_dragons} 只）...")
            self._ai_analyze_dragons(result, top_n_dragons)
        else:
            logger.info("[Step 5] 跳过 AI 龙头分析")

        # 6. 生成策略建议
        logger.info("[Step 6] 生成策略建议...")
        self._generate_advice(result)

        logger.info("=" * 50)
        logger.info("擒龙分析完成")
        logger.info("=" * 50)

        return result

    def _analyze_sentiment(self) -> SentimentData:
        """
        分析市场情绪

        数据来源：
        - ak.stock_zt_pool_em() 涨停池
        - ak.stock_zt_pool_dtgc_em() 跌停池
        - ak.stock_zt_pool_zbgc_em() 炸板池
        - ak.stock_zt_pool_strong_em() 强势股池
        """
        import akshare as ak

        today = datetime.now().strftime("%Y-%m-%d")
        sentiment = SentimentData(date=today)

        # 1. 获取涨停池数据
        limit_up_df = self._call_api_with_retry(lambda: ak.stock_zt_pool_em(date=today.replace("-", "")), "涨停池")

        if limit_up_df is not None and not limit_up_df.empty:
            # 排除 ST 股
            if "名称" in limit_up_df.columns:
                limit_up_df = limit_up_df[~limit_up_df["名称"].str.contains("ST|退", na=False)]

            sentiment.limit_up_count = len(limit_up_df)
            sentiment.real_limit_up = len(limit_up_df)

            # 分析连板情况
            if "连板数" in limit_up_df.columns:
                board_counts = limit_up_df["连板数"].value_counts().to_dict()
                sentiment.continuous_boards = {int(k): int(v) for k, v in board_counts.items()}

                # 最高板
                sentiment.highest_board = int(limit_up_df["连板数"].max())
                highest_stocks = limit_up_df[limit_up_df["连板数"] == sentiment.highest_board]["名称"].tolist()
                sentiment.highest_board_stocks = highest_stocks[:5]  # 最多显示5只

                # 首板数量
                sentiment.first_board_count = sentiment.continuous_boards.get(1, 0)

                # 连板梯队
                ladder_parts = []
                for board in sorted(sentiment.continuous_boards.keys()):
                    count = sentiment.continuous_boards[board]
                    ladder_parts.append(f"{board}板->{count}只")
                sentiment.board_ladder = ", ".join(ladder_parts)

            logger.info(f"[情绪] 涨停: {sentiment.limit_up_count}家, 最高板: {sentiment.highest_board}板")
            logger.info(f"[情绪] 连板梯队: {sentiment.board_ladder}")

        # 2. 获取跌停池数据
        limit_down_df = self._call_api_with_retry(
            lambda: ak.stock_zt_pool_dtgc_em(date=today.replace("-", "")), "跌停池"
        )

        if limit_down_df is not None and not limit_down_df.empty:
            if "名称" in limit_down_df.columns:
                limit_down_df = limit_down_df[~limit_down_df["名称"].str.contains("ST|退", na=False)]
            sentiment.limit_down_count = len(limit_down_df)
            logger.info(f"[情绪] 跌停: {sentiment.limit_down_count}家")

        # 3. 获取炸板池数据
        broken_df = self._call_api_with_retry(lambda: ak.stock_zt_pool_zbgc_em(date=today.replace("-", "")), "炸板池")

        if broken_df is not None and not broken_df_empty(broken_df):
            sentiment.limit_up_broken = len(broken_df)
            # 计算封板率
            total_touched = sentiment.real_limit_up + sentiment.limit_up_broken
            if total_touched > 0:
                sentiment.first_board_success_rate = sentiment.real_limit_up / total_touched * 100
            logger.info(
                f"[情绪] 炸板: {sentiment.limit_up_broken}家, 封板率: {sentiment.first_board_success_rate:.1f}%"
            )

        # 4. 判断情绪周期
        self._judge_sentiment(sentiment)

        return sentiment

    def _judge_sentiment(self, sentiment: SentimentData):
        """
        判断市场情绪周期

        判断标准：
        - 冰点期：涨停<30，跌停>涨停，最高板≤2
        - 回暖期：涨停30-60，连板股出现
        - 发酵期：涨停60-100，连板梯队完整
        - 高潮期：涨停>100，最高板≥5
        - 退潮期：涨停数骤降，高位股分歧
        """
        score = 0
        reasons = []

        up_count = sentiment.limit_up_count
        down_count = sentiment.limit_down_count
        highest = sentiment.highest_board

        # 涨停数量评分（40分）
        if up_count >= 100:
            score += 40
            reasons.append(f"涨停{up_count}家,市场亢奋")
        elif up_count >= 60:
            score += 30
            reasons.append(f"涨停{up_count}家,情绪较好")
        elif up_count >= 30:
            score += 20
            reasons.append(f"涨停{up_count}家,情绪回暖")
        elif up_count >= 15:
            score += 10
            reasons.append(f"涨停{up_count}家,情绪偏弱")
        else:
            score += 0
            reasons.append(f"涨停{up_count}家,市场冰点")

        # 跌停对比评分（20分）
        if up_count > 0:
            ratio = down_count / up_count if up_count > 0 else 999
            if ratio < 0.1:
                score += 20
                reasons.append("跌停极少,多头完胜")
            elif ratio < 0.3:
                score += 15
                reasons.append("涨多跌少,多头占优")
            elif ratio < 0.5:
                score += 10
            elif ratio < 1:
                score += 5
                reasons.append("涨跌接近,多空平衡")
            else:
                score += 0
                reasons.append("跌停多于涨停,空头主导")

        # 连板高度评分（25分）
        if highest >= 7:
            score += 25
            reasons.append(f"最高{highest}板,空间打开")
        elif highest >= 5:
            score += 20
            reasons.append(f"最高{highest}板,高度尚可")
        elif highest >= 3:
            score += 12
            reasons.append(f"最高{highest}板,梯队形成")
        elif highest >= 2:
            score += 5
            reasons.append(f"最高{highest}板,刚起步")
        else:
            score += 0
            reasons.append("无连板,情绪冰点")

        # 封板率评分（15分）
        success_rate = sentiment.first_board_success_rate
        if success_rate >= 80:
            score += 15
            reasons.append(f"封板率{success_rate:.0f}%,封板意愿强")
        elif success_rate >= 60:
            score += 10
            reasons.append(f"封板率{success_rate:.0f}%")
        elif success_rate >= 40:
            score += 5
        else:
            score += 0
            reasons.append(f"封板率仅{success_rate:.0f}%,分歧大")

        sentiment.sentiment_score = score

        # 判断情绪周期
        if score >= 80:
            sentiment.sentiment = MarketSentiment.CLIMAX
            sentiment.sentiment_desc = "市场情绪高潮,注意控制仓位,警惕分歧回落"
        elif score >= 60:
            sentiment.sentiment = MarketSentiment.FERMENTING
            sentiment.sentiment_desc = "情绪发酵中,可积极参与强势板块龙头"
        elif score >= 40:
            sentiment.sentiment = MarketSentiment.WARMING
            sentiment.sentiment_desc = "情绪回暖,可试探性参与,注意仓位"
        elif score >= 20:
            sentiment.sentiment = MarketSentiment.EBBING
            sentiment.sentiment_desc = "情绪退潮,控制仓位,等待企稳"
        else:
            sentiment.sentiment = MarketSentiment.ICE_POINT
            sentiment.sentiment_desc = "市场冰点,观望为主,等待转机"

        logger.info(f"[情绪] 情绪周期: {sentiment.sentiment.value}, 得分: {score}")
        logger.info(f"[情绪] {sentiment.sentiment_desc}")

    def _ai_analyze_sentiment(self, sentiment: SentimentData) -> Optional[Dict[str, Any]]:
        """
        使用 Gemini 进行情绪深度分析

        Args:
            sentiment: 情绪数据

        Returns:
            AI 分析结果字典
        """
        analyzer = self._get_analyzer()
        if not analyzer:
            return None

        try:
            import json

            # 构建 Prompt
            prompt = SENTIMENT_ANALYSIS_PROMPT.format(
                limit_up_count=sentiment.limit_up_count,
                limit_down_count=sentiment.limit_down_count,
                limit_up_broken=sentiment.limit_up_broken,
                success_rate=sentiment.first_board_success_rate,
                highest_board=sentiment.highest_board,
                highest_stocks=", ".join(sentiment.highest_board_stocks[:5]),
                board_ladder=sentiment.board_ladder,
                first_board_count=sentiment.first_board_count,
            )

            logger.info("[AI情绪] 调用 Gemini 分析情绪...")

            # 调用 API
            generation_config = {"temperature": 0.7, "max_output_tokens": 4096}
            response_text = analyzer._call_api_with_retry(prompt, generation_config)

            if not response_text:
                logger.warning("[AI情绪] Gemini 返回空响应")
                return None

            logger.info(f"[AI情绪] 响应长度: {len(response_text)}")
            logger.info(f"[AI情绪] 响应预览: {response_text[:300]}...")

            # 解析 JSON
            cleaned_text = response_text
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
            elif "```" in cleaned_text:
                cleaned_text = cleaned_text.replace("```", "")

            # 使用更好的 JSON 提取方法
            json_str = self._extract_json_from_text(cleaned_text)
            if not json_str:
                logger.warning("[AI情绪] 无法从响应中提取 JSON")
                logger.info(f"[AI情绪] 清理后响应:\n{cleaned_text}")
                return None

            logger.info(f"[AI情绪] 提取的 JSON 长度: {len(json_str)}")

            # 修复 JSON 格式问题
            json_str = self._fix_json_string(json_str)

            try:
                result = json.loads(json_str)
                logger.info(f"[AI情绪] 分析完成: {result.get('sentiment_cycle', '未知')}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"[AI情绪] JSON 解析失败: {e}")
                # 显示错误位置附近内容
                pos = e.pos if hasattr(e, "pos") else 0
                start = max(0, pos - 30)
                end = min(len(json_str), pos + 30)
                logger.info(f"[AI情绪] 错误位置附近: ...{json_str[start:end]}...")
                return None

        except Exception as e:
            logger.error(f"[AI情绪] 分析失败: {e}")
            return None

    def _ai_analyze_dragons(self, result: DragonHuntResult, top_n: int = 5):
        """
        使用 Gemini 对龙头股进行深度分析

        Args:
            result: 擒龙结果
            top_n: 分析前 N 只龙头股
        """
        analyzer = self._get_analyzer()
        if not analyzer:
            return

        # 获取趋势分析器和数据获取器
        trend_analyzer = self._get_trend_analyzer()
        data_fetcher = self._get_data_fetcher()

        # 取评分最高的前 N 只
        top_dragons = sorted(result.dragon_candidates, key=lambda x: x.total_score, reverse=True)[:top_n]

        for i, dragon in enumerate(top_dragons):
            try:
                logger.info(f"[AI龙头] ({i+1}/{len(top_dragons)}) 分析 {dragon.name}({dragon.code})...")

                # 获取股票历史数据用于技术分析
                technical_data = ""
                if data_fetcher and trend_analyzer:
                    try:
                        df = data_fetcher.get_daily_data(dragon.code, days=30)
                        if df is not None and not df.empty:
                            trend_result = trend_analyzer.analyze(df, dragon.code)
                            technical_data = f"""
| 指标 | 数值 |
|------|------|
| 趋势状态 | {trend_result.trend_status.value} |
| 均线排列 | {trend_result.ma_alignment} |
| 趋势强度 | {trend_result.trend_strength}/100 |
| MA5乖离率 | {trend_result.bias_ma5:+.2f}% |
| 量能状态 | {trend_result.volume_status.value} |
| 系统信号 | {trend_result.buy_signal.value} |
| 系统评分 | {trend_result.signal_score}/100 |
"""
                            # 更新龙头的技术指标
                            dragon.ma_trend = trend_result.ma_alignment
                            dragon.volume_pattern = trend_result.volume_trend
                    except Exception as e:
                        logger.warning(f"[AI龙头] 获取 {dragon.code} 技术数据失败: {e}")

                # 构建 Prompt
                prompt = DRAGON_ANALYSIS_PROMPT.format(
                    sentiment_cycle=result.sentiment.sentiment.value if result.sentiment else "未知",
                    sentiment_score=result.sentiment.sentiment_score if result.sentiment else 50,
                    highest_board=result.sentiment.highest_board if result.sentiment else 0,
                    stock_name=dragon.name,
                    stock_code=dragon.code,
                    change_pct=dragon.change_pct,
                    continuous_boards=dragon.continuous_boards,
                    turnover_rate=dragon.turnover_rate,
                    amount=dragon.amount,
                    circ_mv=dragon.circ_mv,
                    catalyst=dragon.catalyst or "未知",
                    dragon_level=dragon.dragon_level.value,
                    technical_data=technical_data if technical_data else "暂无技术面数据",
                )

                # 调用 API（使用最大 token 限制，避免响应被截断）
                generation_config = {"temperature": 0.7, "max_output_tokens": 8192}
                response_text = analyzer._call_api_with_retry(prompt, generation_config)

                if response_text:
                    logger.info(f"[AI龙头] {dragon.name} 响应长度: {len(response_text)}")
                    # 打印响应的前200字符帮助调试
                    logger.info(f"[AI龙头] {dragon.name} 响应预览: {response_text[:200]}...")
                    ai_result = self._parse_dragon_ai_response(response_text, dragon.code, dragon.name)
                    if ai_result:
                        dragon.ai_analysis = ai_result
                        logger.info(f"[AI龙头] {dragon.name}: {ai_result.action}, 评分 {ai_result.comprehensive_score}")
                    else:
                        logger.warning(f"[AI龙头] {dragon.name} 解析失败，跳过")
                        # 保存失败的响应到文件便于调试
                        try:
                            debug_file = f"debug_dragon_{dragon.code}.txt"
                            with open(debug_file, "w", encoding="utf-8") as f:
                                f.write(response_text)
                            logger.info(f"[AI龙头] 已保存响应到 {debug_file}")
                        except Exception:
                            pass
                else:
                    logger.warning(f"[AI龙头] {dragon.name} API 返回空响应")

                # 请求间隔，避免限流
                time.sleep(2)

            except Exception as e:
                logger.error(f"[AI龙头] 分析 {dragon.name} 失败: {e}")

    def _try_complete_json(self, json_str: str) -> str:
        """尝试修复不完整的 JSON（添加缺失的闭合括号）"""
        # 统计括号
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape = False

        for char in json_str:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1

        # 如果有未闭合的括号，尝试修复
        if brace_count > 0 or bracket_count > 0:
            logger.warning(f"[JSON修复] 检测到不完整 JSON: 缺少 {brace_count} 个 '}}' 和 {bracket_count} 个 ']'")

            # 检查最后是否在字符串中间被截断
            # 如果最后几个字符看起来像是被截断的字符串，尝试关闭它
            stripped = json_str.rstrip()
            if stripped and stripped[-1] not in '",}]':
                # 可能在字符串中间截断，尝试关闭字符串
                json_str = stripped + '"'
                logger.info("[JSON修复] 添加了缺失的引号")

            # 添加缺失的闭合括号
            json_str = json_str + "]" * bracket_count + "}" * brace_count
            logger.info(f"[JSON修复] 已添加 {bracket_count} 个 ']' 和 {brace_count} 个 '}}'")

        return json_str

    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        # 移除注释
        json_str = re.sub(r"//.*?\n", "\n", json_str)
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

        # 移除控制字符（保留换行和空格）
        json_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", json_str)

        # 将换行符替换为空格（JSON 字符串内不能有换行）
        # 但要小心不要破坏字符串外的结构
        json_str = json_str.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

        # 修复多余的空格
        json_str = re.sub(r"\s+", " ", json_str)

        # 修复尾随逗号
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # 确保布尔值是小写
        json_str = json_str.replace("True", "true").replace("False", "false")
        json_str = json_str.replace("None", "null")

        # 修复字符串中的未转义引号（常见问题）
        # 例如: "value with "quotes" inside" -> "value with \"quotes\" inside"
        # 这个比较复杂，用简单方法：替换中文引号
        json_str = json_str.replace(""", '"').replace(""", '"')
        json_str = json_str.replace("'", "'").replace("'", "'")

        return json_str

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 对象，处理嵌套括号和 markdown 代码块"""
        # 先移除 markdown 代码块标记
        cleaned = text
        if "```json" in cleaned:
            cleaned = cleaned.replace("```json", "")
        if "```" in cleaned:
            cleaned = cleaned.replace("```", "")

        # 找到第一个 {
        start = cleaned.find("{")
        if start == -1:
            return None

        # 使用括号计数找到匹配的 }
        count = 0
        in_string = False
        escape = False
        end = start

        for i, char in enumerate(cleaned[start:], start):
            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                count += 1
            elif char == "}":
                count -= 1
                if count == 0:
                    end = i + 1
                    break

        if count != 0:
            # 括号不匹配（可能响应被截断），尝试找最后一个 }
            last_brace = cleaned.rfind("}")
            if last_brace > start:
                end = last_brace + 1
                logger.warning(f"[JSON提取] 括号不匹配，尝试使用最后的 '}}' 位置: {end}")
            else:
                logger.warning(f"[JSON提取] JSON 不完整，缺少闭合括号")
                return None

        if end > start:
            return cleaned[start:end]

        return None

    def _parse_dragon_ai_response(self, response_text: str, code: str, name: str) -> Optional[DragonAIAnalysis]:
        """解析龙头股 AI 分析响应"""
        try:
            import json

            # 清理响应
            cleaned_text = response_text
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
            elif "```" in cleaned_text:
                cleaned_text = cleaned_text.replace("```", "")

            # 提取 JSON
            json_str = self._extract_json_from_text(cleaned_text)
            if not json_str:
                logger.warning(f"[AI龙头] 无法从响应中提取 JSON（未找到有效的 {{...}}）")
                logger.info(f"[AI龙头] 原始响应:\n{response_text}")
                return None

            logger.info(f"[AI龙头] 提取的 JSON 长度: {len(json_str)}")

            # 修复 JSON 格式问题
            json_str = self._fix_json_string(json_str)
            logger.info(f"[AI龙头] 修复后 JSON 长度: {len(json_str)}")

            # 检查 JSON 是否完整（括号是否匹配）
            json_str = self._try_complete_json(json_str)

            data = None
            parse_error = None

            # 尝试解析
            try:
                data = json.loads(json_str)
                logger.info(f"[AI龙头] JSON 解析成功！stock_code={data.get('stock_code')}")
            except json.JSONDecodeError as e:
                parse_error = e
                logger.warning(f"[AI龙头] JSON 解析失败 (第一次): {e}")
                # 输出错误位置附近的内容
                pos = e.pos if hasattr(e, "pos") else 0
                start = max(0, pos - 50)
                end = min(len(json_str), pos + 50)
                logger.info(f"[AI龙头] 第一次错误位置 {pos}, 附近: ...{repr(json_str[start:end])}...")

                # 尝试更激进的修复：移除可能有问题的字符
                try:
                    # 尝试移除字符串值中的特殊字符
                    fixed_str = re.sub(
                        r':\s*"([^"]*)"',
                        lambda m: ': "' + m.group(1).replace("\\", "\\\\").replace('"', '\\"') + '"',
                        json_str,
                    )
                    data = json.loads(fixed_str)
                    logger.info(f"[AI龙头] 修复后解析成功")
                except json.JSONDecodeError:
                    pass

            if data is None:
                # 最后尝试：使用更宽松的解析
                try:
                    # 移除所有可能导致问题的字符
                    simple_str = re.sub(r'[^\x20-\x7e{}[\]:,"]', "", json_str)
                    data = json.loads(simple_str)
                    logger.info(f"[AI龙头] 简化后解析成功")
                except json.JSONDecodeError as e2:
                    logger.warning(f"[AI龙头] JSON 最终解析失败: {e2}")
                    # 输出问题位置附近的内容
                    if parse_error:
                        pos = parse_error.pos if hasattr(parse_error, "pos") else 0
                        start = max(0, pos - 50)
                        end = min(len(json_str), pos + 50)
                        logger.info(f"[AI龙头] 错误位置附近内容: ...{json_str[start:end]}...")
                    return None

            # 检查是否有必需的字段（dragon_assessment）来判断是否使用了正确的格式
            if "dragon_assessment" not in data:
                logger.warning(f"[AI龙头] 响应格式不正确，缺少 dragon_assessment 字段")
                logger.info(f"[AI龙头] 返回的字段: {list(data.keys())}")
                return None

            # 解析各字段
            dragon_assess = data.get("dragon_assessment", {})
            capital = data.get("capital_analysis", {})
            fundamental = data.get("fundamental_check", {})
            position = data.get("position_analysis", {})
            space = data.get("space_expectation", {})
            operation = data.get("operation_suggestion", {})

            # 处理 comprehensive_score 可能是字符串或数字的情况
            score = data.get("comprehensive_score", 0)
            if isinstance(score, str):
                try:
                    score = int(float(score.replace("%", "")))
                except (ValueError, TypeError):
                    score = 50

            return DragonAIAnalysis(
                code=code,
                name=name,
                recognition_level=str(dragon_assess.get("recognition_level", "")),
                is_true_dragon=bool(dragon_assess.get("is_true_dragon", False)),
                sector_position=str(dragon_assess.get("sector_position", "")),
                turnover_status=str(capital.get("turnover_status", "")),
                main_force_action=str(capital.get("main_force_action", "")),
                catalyst_quality=str(fundamental.get("catalyst_quality", "")),
                sustainability=str(fundamental.get("sustainability", "")),
                current_position=str(position.get("current_position", "")),
                entry_safety=str(position.get("entry_safety", "")),
                support_level=str(position.get("support_level", "")),
                pressure_level=str(position.get("pressure_level", "")),
                upside_space=str(space.get("upside_space", "")),
                target_price=str(space.get("target_price", "")),
                action=str(operation.get("action", "")),
                entry_timing=str(operation.get("entry_timing", "")),
                position_size=str(operation.get("position_size", "")),
                stop_loss=str(operation.get("stop_loss", "")),
                comprehensive_score=score,
                one_sentence_conclusion=str(data.get("one_sentence_conclusion", "")),
                risk_warning=data.get("risk_warning", []) if isinstance(data.get("risk_warning"), list) else [],
                raw_analysis=data,
            )

        except Exception as e:
            logger.warning(f"[AI龙头] 解析响应失败: {e}")
            import traceback

            logger.info(f"[AI龙头] 详细错误:\n{traceback.format_exc()}")

        return None

    def _analyze_sectors(self) -> List[SectorStrength]:
        """
        分析强势板块

        筛选标准：
        1. 涨幅靠前
        2. 与大盘共振（大盘涨则板块涨更多）
        3. 有涨停股带动
        4. 主力资金流入
        """
        import akshare as ak

        strong_sectors = []

        # 1. 获取行业板块行情
        sector_df = self._call_api_with_retry(ak.stock_board_industry_name_em, "行业板块")

        if sector_df is None or sector_df.empty:
            logger.warning("[板块] 获取板块数据失败")
            return strong_sectors

        # 获取大盘涨跌幅（上证指数）
        index_df = self._call_api_with_retry(ak.stock_zh_index_spot_sina, "指数行情")
        market_change = 0.0
        if index_df is not None and not index_df.empty:
            sh_row = index_df[index_df["代码"] == "sh000001"]
            if not sh_row.empty:
                market_change = self._safe_float(sh_row.iloc[0].get("涨跌幅", 0))

        logger.info(f"[板块] 大盘涨跌幅: {market_change:.2f}%")

        # 筛选强势板块（涨幅前20）
        if "涨跌幅" in sector_df.columns:
            sector_df["涨跌幅"] = pd.to_numeric(sector_df["涨跌幅"], errors="coerce")
            sector_df = sector_df.dropna(subset=["涨跌幅"])
            top_sectors = sector_df.nlargest(20, "涨跌幅")

            for _, row in top_sectors.iterrows():
                name = str(row.get("板块名称", ""))
                change = self._safe_float(row.get("涨跌幅"))

                sector = SectorStrength(
                    name=name,
                    change_pct=change,
                    up_count=self._safe_int(row.get("上涨家数", 0)),
                    down_count=self._safe_int(row.get("下跌家数", 0)),
                )

                # 判断是否与大盘共振
                # 大盘涨时板块涨更多，或大盘跌时板块抗跌
                if market_change > 0:
                    sector.is_resonating = change > market_change
                else:
                    sector.is_resonating = change > 0 or change > market_change * 0.5

                # 只保留共振板块或涨幅>2%的板块
                if sector.is_resonating or change >= 2.0:
                    strong_sectors.append(sector)
                    logger.info(f"[板块] {name}: {change:+.2f}% {'(共振)' if sector.is_resonating else ''}")

        # 限制返回数量
        return strong_sectors[:10]

    def _hunt_dragons(self, sentiment: SentimentData, sectors: List[SectorStrength]) -> List[DragonCandidate]:
        """
        猎龙：在强势板块中寻找龙头股

        筛选标准：
        1. 涨停或大涨（>5%）
        2. 换手充分（5%-15%最佳）
        3. 成交额适中（1-20亿）
        4. 有业绩/预期支撑
        """
        import akshare as ak

        candidates = []
        sector_names = [s.name for s in sectors[:5]]  # 取前5个强势板块

        logger.info(f"[猎龙] 目标板块: {sector_names}")

        # 1. 获取涨停池的股票作为龙头候选
        limit_up_df = self._call_api_with_retry(
            lambda: ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d")), "涨停池"
        )

        if limit_up_df is not None and not limit_up_df.empty:
            # 排除 ST
            if "名称" in limit_up_df.columns:
                limit_up_df = limit_up_df[~limit_up_df["名称"].str.contains("ST|退", na=False)]

            for _, row in limit_up_df.iterrows():
                code = str(row.get("代码", ""))
                name = str(row.get("名称", ""))

                candidate = DragonCandidate(
                    code=code,
                    name=name,
                    change_pct=self._safe_float(row.get("涨跌幅", 10.0)),
                    turnover_rate=self._safe_float(row.get("换手率")),
                    amount=self._safe_float(row.get("成交额")) / 1e8,  # 转亿
                    circ_mv=self._safe_float(row.get("流通市值")) / 1e8,  # 转亿
                    continuous_boards=self._safe_int(row.get("连板数", 1)),
                )

                # 涨停原因/所属板块
                if "涨停原因" in limit_up_df.columns:
                    candidate.sector = str(row.get("涨停原因", ""))
                    candidate.catalyst = candidate.sector

                # 判断龙头级别
                if candidate.continuous_boards >= sentiment.highest_board and sentiment.highest_board >= 3:
                    candidate.dragon_level = DragonLevel.SPACE_DRAGON
                elif candidate.continuous_boards >= 2:
                    candidate.dragon_level = DragonLevel.SECTOR_DRAGON
                else:
                    candidate.dragon_level = DragonLevel.FOLLOWER

                # 评分
                self._score_candidate(candidate, sentiment)

                candidates.append(candidate)

        # 2. 获取板块成分股中的强势股
        for sector in sectors[:3]:  # 取前3个板块深入分析
            sector_stocks = self._get_sector_leaders(sector.name)
            for stock in sector_stocks:
                # 检查是否已在候选列表中
                if not any(c.code == stock.code for c in candidates):
                    stock.sector = sector.name
                    self._score_candidate(stock, sentiment)
                    candidates.append(stock)

        # 按评分排序
        candidates.sort(key=lambda x: x.total_score, reverse=True)

        # 返回前15个候选
        return candidates[:15]

    def _get_sector_leaders(self, sector_name: str) -> List[DragonCandidate]:
        """获取板块领涨股"""
        import akshare as ak

        leaders = []

        try:
            # 获取板块成分股
            df = self._call_api_with_retry(
                lambda: ak.stock_board_industry_cons_em(symbol=sector_name), f"板块成分-{sector_name}"
            )

            if df is not None and not df.empty:
                # 排除 ST
                if "名称" in df.columns:
                    df = df[~df["名称"].str.contains("ST|退", na=False)]

                # 筛选涨幅>3%的股票
                if "涨跌幅" in df.columns:
                    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
                    strong_df = df[df["涨跌幅"] >= 3].nlargest(5, "涨跌幅")

                    for _, row in strong_df.iterrows():
                        candidate = DragonCandidate(
                            code=str(row.get("代码", "")),
                            name=str(row.get("名称", "")),
                            change_pct=self._safe_float(row.get("涨跌幅")),
                            turnover_rate=self._safe_float(row.get("换手率")),
                            amount=self._safe_float(row.get("成交额")) / 1e8,
                            circ_mv=self._safe_float(row.get("流通市值")) / 1e8,
                        )
                        leaders.append(candidate)

        except Exception as e:
            logger.warning(f"[猎龙] 获取板块 {sector_name} 成分股失败: {e}")

        return leaders

    def _score_candidate(self, candidate: DragonCandidate, sentiment: SentimentData):
        """
        对龙头候选股评分

        评分维度（满分100）：
        1. 连板高度（20分）：连板越高，辨识度越强
        2. 换手率（20分）：5%-15%为最佳区间
        3. 成交额（15分）：1-10亿最佳
        4. 流通市值（15分）：20-100亿最佳
        5. 板块地位（15分）：是否板块龙头
        6. 业绩预期（15分）：有业绩/预期支撑
        """
        score = 0
        details = {}

        # 1. 连板高度（20分）
        boards = candidate.continuous_boards
        if boards >= 5:
            board_score = 20
        elif boards >= 3:
            board_score = 15
        elif boards >= 2:
            board_score = 10
        elif boards >= 1 and candidate.change_pct >= 9.5:  # 首板涨停
            board_score = 8
        else:
            board_score = 3
        score += board_score
        details["连板高度"] = board_score

        # 2. 换手率（20分）
        turnover = candidate.turnover_rate
        if 5 <= turnover <= 15:
            turnover_score = 20  # 最佳区间
        elif 3 <= turnover < 5 or 15 < turnover <= 20:
            turnover_score = 15  # 可接受
        elif 1 <= turnover < 3 or 20 < turnover <= 30:
            turnover_score = 8  # 偏低/偏高
        else:
            turnover_score = 3  # 不理想
        score += turnover_score
        details["换手率"] = turnover_score

        # 3. 成交额（15分）
        amount = candidate.amount
        if 1 <= amount <= 10:
            amount_score = 15  # 最佳
        elif 0.5 <= amount < 1 or 10 < amount <= 20:
            amount_score = 10  # 可接受
        elif amount > 20:
            amount_score = 6  # 大资金关注但可能分歧
        else:
            amount_score = 3  # 流动性不足
        score += amount_score
        details["成交额"] = amount_score

        # 4. 流通市值（15分）- 偏好中小市值
        circ_mv = candidate.circ_mv
        if 20 <= circ_mv <= 100:
            mv_score = 15  # 最佳
        elif 10 <= circ_mv < 20 or 100 < circ_mv <= 200:
            mv_score = 10  # 可接受
        elif circ_mv < 10:
            mv_score = 8  # 小盘股波动大
        else:
            mv_score = 5  # 大盘股弹性小
        score += mv_score
        details["流通市值"] = mv_score

        # 5. 龙头地位（15分）
        if candidate.dragon_level == DragonLevel.SPACE_DRAGON:
            level_score = 15
            candidate.is_sector_first = True
        elif candidate.dragon_level == DragonLevel.SECTOR_DRAGON:
            level_score = 12
            candidate.is_sector_first = True
        elif candidate.dragon_level == DragonLevel.FOLLOWER:
            level_score = 6
        else:
            level_score = 2
        score += level_score
        details["龙头地位"] = level_score

        # 6. 业绩预期（15分）- 基于涨停原因关键词判断
        catalyst = candidate.catalyst.lower() if candidate.catalyst else ""
        performance_keywords = ["业绩", "利润", "营收", "订单", "中标", "合同", "收购", "并购"]
        expectation_keywords = ["政策", "利好", "规划", "突破", "新产品", "技术", "专利", "ai", "机器人"]

        has_perf = any(kw in catalyst for kw in performance_keywords)
        has_expect = any(kw in catalyst for kw in expectation_keywords)

        candidate.has_performance = has_perf
        candidate.has_expectation = has_expect

        if has_perf and has_expect:
            expect_score = 15  # 业绩+预期
        elif has_perf:
            expect_score = 12  # 纯业绩
        elif has_expect:
            expect_score = 10  # 有预期
        else:
            expect_score = 3  # 纯情绪
        score += expect_score
        details["业绩预期"] = expect_score

        # 汇总
        candidate.total_score = score
        candidate.score_details = details

        # 生成买入信号
        if score >= 80:
            candidate.buy_signal = "强烈关注"
        elif score >= 65:
            candidate.buy_signal = "建议关注"
        elif score >= 50:
            candidate.buy_signal = "可以观察"
        else:
            candidate.buy_signal = "暂时观望"

        # 风险提示
        risks = []
        if candidate.continuous_boards >= 4:
            risks.append("高位股,注意追高风险")
        if turnover > 25:
            risks.append("换手过高,筹码松动")
        if amount > 30:
            risks.append("成交过大,可能有分歧")
        if not has_perf and not has_expect:
            risks.append("缺乏业绩/预期支撑")

        candidate.risk_warning = "; ".join(risks) if risks else "风险可控"

    def _generate_advice(self, result: DragonHuntResult):
        """生成策略建议"""
        sentiment = result.sentiment
        dragons = result.dragon_candidates
        sectors = result.strong_sectors

        advice_parts = []

        # 1. 情绪判断
        if sentiment is None:
            advice_parts.append("无法获取市场情绪数据，建议观望")
            result.position_advice = "建议仓位：1-2成（谨慎）"
        elif sentiment.sentiment == MarketSentiment.ICE_POINT:
            advice_parts.append("当前市场处于冰点期，建议空仓观望，等待情绪修复信号（涨停数回升、连板股出现）")
            result.position_advice = "建议仓位：0-2成"
        elif sentiment.sentiment == MarketSentiment.WARMING:
            advice_parts.append("情绪回暖中，可小仓位试探强势板块龙头首板，严格止损")
            result.position_advice = "建议仓位：2-4成"
        elif sentiment.sentiment == MarketSentiment.FERMENTING:
            advice_parts.append("情绪发酵期，可积极参与连板龙头和板块首板，跟随主线")
            result.position_advice = "建议仓位：4-6成"
        elif sentiment.sentiment == MarketSentiment.CLIMAX:
            advice_parts.append("情绪高潮期，注意控制仓位，警惕分歧回落，高位股谨慎追涨")
            result.position_advice = "建议仓位：3-5成（控制风险）"
        else:  # EBBING
            advice_parts.append("情绪退潮中，降低仓位，等待新主线确认")
            result.position_advice = "建议仓位：1-3成"

        # 2. 板块建议
        if sectors:
            resonating = [s for s in sectors if s.is_resonating][:3]
            if resonating:
                names = [s.name for s in resonating]
                advice_parts.append(f"与大盘共振的强势板块：{', '.join(names)}，可重点关注")

        # 3. 个股建议
        top_dragons = [d for d in dragons if d.total_score >= 65][:5]
        if top_dragons:
            advice_parts.append("重点关注的龙头候选：")
            for d in top_dragons:
                level_tag = f"[{d.dragon_level.value}]" if d.dragon_level != DragonLevel.WEAK else ""
                perf_tag = "[有业绩]" if d.has_performance else ""
                expect_tag = "[有预期]" if d.has_expectation else ""
                advice_parts.append(f"  - {d.name}({d.code}) {level_tag}{perf_tag}{expect_tag} 评分:{d.total_score}")

        # 4. 风格提示
        advice_parts.append("\n当前市场风格偏向有业绩、有预期的股票，纯情绪炒作需谨慎")

        result.strategy_advice = "\n".join(advice_parts)

    def format_report(self, result: DragonHuntResult) -> str:
        """
        格式化输出报告

        Args:
            result: 擒龙分析结果

        Returns:
            格式化的报告文本
        """
        lines = []

        # 标题
        lines.append(f"## 🐉 {result.date} 短线擒龙报告")
        if result.ai_enabled:
            lines.append("*（已启用 AI 深度分析）*")
        lines.append("")

        # 情绪分析
        s = result.sentiment
        lines.append("### 一、市场情绪分析")
        if s:
            lines.append(f"- **情绪周期**: {s.sentiment.value} (得分: {s.sentiment_score}/100)")
            lines.append(f"- **涨停家数**: {s.limit_up_count}家")
            lines.append(f"- **跌停家数**: {s.limit_down_count}家")
            lines.append(f"- **最高连板**: {s.highest_board}板 ({', '.join(s.highest_board_stocks[:3])})")
            lines.append(f"- **连板梯队**: {s.board_ladder}")
            if s.first_board_success_rate > 0:
                lines.append(f"- **封板率**: {s.first_board_success_rate:.1f}%")
            lines.append(f"- **情绪研判**: {s.sentiment_desc}")

            # AI 情绪分析结果
            if result.ai_sentiment_analysis:
                ai = result.ai_sentiment_analysis
                lines.append("")
                lines.append("#### 🤖 AI 情绪深度分析")
                core = ai.get("core_analysis", {})
                if core:
                    lines.append(f"- **情绪状态**: {core.get('emotion_status', '')}")
                    lines.append(f"- **市场主线**: {core.get('main_line', '')}")
                    lines.append(f"- **资金流向**: {core.get('money_flow', '')}")
                    lines.append(f"- **风险等级**: {core.get('risk_level', '')}")

                ladder = ai.get("ladder_analysis", {})
                if ladder:
                    lines.append(f"- **空间高度**: {ladder.get('space_height', '')}")
                    lines.append(f"- **梯队健康**: {ladder.get('ladder_health', '')}")

                op = ai.get("operation_advice", {})
                if op:
                    lines.append(f"- **策略建议**: {op.get('strategy', '')}")
                    lines.append(f"- **关注方向**: {op.get('focus_direction', '')}")

                tomorrow = ai.get("tomorrow_outlook", {})
                if tomorrow:
                    lines.append(f"- **明日展望**: {tomorrow.get('emotion_trend', '')}")

                risks = ai.get("risk_warning", [])
                if risks:
                    lines.append(f"- **风险提示**: {'; '.join(risks)}")
        else:
            lines.append("暂无情绪数据")
        lines.append("")

        # 强势板块
        lines.append("### 二、强势板块")
        if result.strong_sectors:
            lines.append("| 板块 | 涨跌幅 | 共振 |")
            lines.append("|------|--------|------|")
            for sector in result.strong_sectors[:8]:
                resonating = "✓" if sector.is_resonating else ""
                lines.append(f"| {sector.name} | {sector.change_pct:+.2f}% | {resonating} |")
        else:
            lines.append("暂无明显强势板块")
        lines.append("")

        # 龙头候选（简表）
        lines.append("### 三、龙头候选")
        if result.dragon_candidates:
            lines.append("| 股票 | 涨幅 | 连板 | 换手 | 评分 | 龙头级别 | 建议 |")
            lines.append("|------|------|------|------|------|----------|------|")
            for d in result.dragon_candidates[:10]:
                level = d.dragon_level.value[:2]  # 简写
                lines.append(
                    f"| {d.name}({d.code}) | {d.change_pct:+.1f}% | {d.continuous_boards}板 | "
                    f"{d.turnover_rate:.1f}% | {d.total_score} | {level} | {d.buy_signal} |"
                )
        else:
            lines.append("暂无符合条件的龙头候选")
        lines.append("")

        # AI 龙头深度分析
        ai_dragons = [d for d in result.dragon_candidates if d.ai_analysis is not None]
        if ai_dragons:
            lines.append("### 四、🤖 龙头股 AI 深度分析")
            lines.append("")
            for d in ai_dragons:
                ai = d.ai_analysis
                if ai is None:  # 类型守卫
                    continue
                lines.append(f"#### {d.name}（{d.code}）")
                lines.append(f"**{ai.one_sentence_conclusion}**")
                lines.append("")
                lines.append(f"- **AI 综合评分**: {ai.comprehensive_score}/100")
                lines.append(f"- **操作建议**: {ai.action}")
                lines.append(f"- **龙头辨识度**: {ai.recognition_level}")
                lines.append(f"- **真龙判定**: {'是 ✓' if ai.is_true_dragon else '否'}")
                lines.append(f"- **换手率状态**: {ai.turnover_status}")
                lines.append(f"- **主力动向**: {ai.main_force_action}")
                lines.append(f"- **当前位置**: {ai.current_position}")
                lines.append(f"- **介入安全度**: {ai.entry_safety}")
                if ai.support_level:
                    lines.append(f"- **支撑位**: {ai.support_level}")
                if ai.pressure_level:
                    lines.append(f"- **压力位**: {ai.pressure_level}")
                lines.append(f"- **上涨空间**: {ai.upside_space}")
                if ai.target_price:
                    lines.append(f"- **目标价位**: {ai.target_price}")
                lines.append(f"- **介入时机**: {ai.entry_timing}")
                lines.append(f"- **建议仓位**: {ai.position_size}")
                if ai.stop_loss:
                    lines.append(f"- **止损位置**: {ai.stop_loss}")
                if ai.risk_warning:
                    lines.append(f"- **风险提示**: {'; '.join(ai.risk_warning)}")
                lines.append("")
            lines.append("")

        # 策略建议
        section_num = "五" if ai_dragons else "四"
        lines.append(f"### {section_num}、策略建议")
        lines.append(result.strategy_advice)
        lines.append("")
        lines.append(f"**{result.position_advice}**")
        lines.append("")

        # 风险提示
        section_num = "六" if ai_dragons else "五"
        lines.append(f"### {section_num}、风险提示")
        lines.append("- 短线交易风险较高，需严格执行止损")
        lines.append("- 追高涨停股需谨慎，注意买在分歧、卖在一致")
        lines.append("- 以上分析仅供参考，不构成投资建议")
        lines.append("")

        lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

        return "\n".join(lines)


def broken_df_empty(df) -> bool:
    """检查 DataFrame 是否为空"""
    return df is None or df.empty


def run_dragon_hunt(enable_ai: bool = True, top_n_dragons: int = 5) -> str:
    """
    执行短线擒龙分析

    Args:
        enable_ai: 是否启用 AI 分析（默认启用）
        top_n_dragons: 对前 N 只龙头股进行 AI 深度分析

    Returns:
        格式化的报告文本
    """
    hunter = DragonHunter(enable_ai=enable_ai)
    result = hunter.analyze(top_n_dragons=top_n_dragons)
    report = hunter.format_report(result)
    return report


def run_dragon_hunt_simple() -> str:
    """
    执行简单版擒龙分析（不使用 AI）

    Returns:
        格式化的报告文本
    """
    return run_dragon_hunt(enable_ai=False)


# 测试入口
if __name__ == "__main__":
    import sys
    import argparse

    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="短线擒龙策略分析")
    parser.add_argument("--no-ai", action="store_true", help="禁用 AI 分析")
    parser.add_argument("--top-n", type=int, default=5, help="AI 分析前 N 只龙头股")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    )

    print("=" * 60)
    print("短线擒龙策略分析")
    print(f"AI 分析: {'禁用' if args.no_ai else '启用'}")
    print(f"龙头分析数量: {args.top_n}")
    print("=" * 60)

    try:
        report = run_dragon_hunt(enable_ai=not args.no_ai, top_n_dragons=args.top_n)
        print("\n" + report)
    except Exception as e:
        logger.error(f"擒龙分析失败: {e}")
        import traceback

        traceback.print_exc()
