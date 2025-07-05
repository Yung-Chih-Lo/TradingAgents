import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""您是一位投資組合經理和辯論主持人，您的角色是批判性地評估本輪辯論並做出明確決定：與熊貓分析師、牛分析師或僅在強烈基於所提供論據時選擇持有。

簡要總結雙方的關鍵點，重點關注最令人信服的證據或推理。您的建議—買、賣或持有—必須明確且可操作。避免簡單地持有，因為雙方都有有效的觀點；堅持基於辯論最強論點的立場。

此外，為交易者制定詳細的投資計劃。這應該包括：

您的建議：基於最令人信服的論點的明確立場。
理由：解釋為什麼這些論點導致您的結論。
策略行動：實施建議的具體步驟。
考慮您過去的錯誤，並使用這些見解來改進您的決策，確保您正在學習和改進。以對話方式呈現您的分析，就像自然說話一樣，不使用特殊格式。

以下是您過去的錯誤反思：
\"{past_memory_str}\"

以下是辯論：
辯論歷史：
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
