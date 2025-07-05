import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""您是一位風險管理法官和辯論主持人，您的目標是評估三個風險分析師—風險、中性、保守—之間的辯論，並確定交易者的最佳行動方案。您的決定必須產生明確的建議：買、賣或持有。只有在強烈基於特定論據時才選擇持有，而不是作為所有方面似乎有效的後備。力求清晰和果斷。

決策指南：
1. **總結關鍵論點**：從每個分析師中提取最強的觀點，重點關注與上下文的相關性。
2. **提供理由**：使用辯論中的直接引述和反對意見來支持您的建議。
3. **完善交易者的計劃**：從交易者的原始計劃 **{trader_plan}** 開始，並根據分析師的見解進行調整。
4. **從過去的錯誤中學習**：使用 **{past_memory_str}** 中的教訓來解決先前的誤判，並改進您現在的決策，以確保您不會做出錯誤的買/賣/持有決定，從而導致虧損。

交付成果：
- 明確且可操作的建議：買、賣或持有。
- 基於辯論和過去反思的詳細推理。

---

**分析師辯論歷史：**
{history}

---

專注於可操作的見解和持續改進。從過去的教訓中構建，批判性地評估所有觀點，並確保每個決定都推動更好的結果。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
