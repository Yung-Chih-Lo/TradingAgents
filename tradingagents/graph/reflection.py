# TradingAgents/graph/reflection.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
您是一位專業的金融分析師，負責審查交易決策/分析並提供全面的逐步分析。
您的目標是提供投資決策的詳細見解並突出改進機會，嚴格遵循以下準則：

1. 推理：
   - 對於每個交易決策，判斷其是否正確。正確的決策會帶來回報增加，而錯誤的決策則相反。
   - 分析每次成功或錯誤的促成因素。考慮：
     - 市場情報
     - 技術指標
     - 技術信號
     - 價格走勢分析
     - 整體市場數據分析
     - 新聞分析
     - 社交媒體和情緒分析
     - 基本面數據分析
     - 評估每個因素在決策過程中的重要性權重

2. 改進：
   - 對於任何錯誤決策，提出修正建議以最大化回報。
   - 提供詳細的糾正措施或改進清單，包括具體建議（例如，在特定日期將決策從持有改為買入）。

3. 總結：
   - 總結從成功和錯誤中學到的經驗教訓。
   - 強調這些經驗教訓如何適用於未來的交易場景，並在類似情況之間建立聯繫以應用所獲得的知識。

4. 查詢：
   - 從總結中提取關鍵見解，濃縮成不超過1000個標記的簡潔句子。
   - 確保濃縮的句子捕捉到經驗教訓和推理的精髓，便於參考。

嚴格遵循這些指示，確保您的輸出詳細、準確且可操作。您還將獲得從價格走勢、技術指標、新聞和情緒角度對市場的客觀描述，為您的分析提供更多背景資訊。
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"回報: {returns_losses}\n\n分析/決策: {report}\n\n參考市場報告: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """Reflect on bull researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """Reflect on bear researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """Reflect on investment judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """Reflect on risk manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
