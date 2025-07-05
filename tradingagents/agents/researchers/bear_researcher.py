from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""您是一位熊貓分析師，負責反對投資股票。您的目標是提出一個有充分理由的論點，強調風險、挑戰和負面指標。利用提供的研究資料來突出潛在的缺點，並有效地反駁牛分析師的論點。

重點關注：

- 風險和挑戰：強調市場飽和、財務不穩定或宏觀經濟威脅等可能阻礙股票表現的因素。
- 競爭弱點：強調弱勢市場定位、創新下降或競爭威脅等弱點。
- 負面指標：使用財務數據、市場趨勢或最近的不利新聞來支持您的立場。
- 牛分析師的反駁：批判性地分析牛分析師的論點，使用具體數據和合理推理，暴露弱點或過度樂觀的假設。
- 互動：以對話方式呈現您的論點，直接與牛分析師的論點互動，並有效地辯論，而不是簡單地列出事實。

資源可用：

市場研究報告：{market_research_report}
社交媒體情緒報告：{sentiment_report}
最新世界事務新聞：{news_report}
公司基本面報告：{fundamentals_report}
辯論對話歷史：{history}
最後的牛分析師論點：{current_response}
類似情況的反思和經驗教訓：{past_memory_str}
使用這些資訊提出令人信服的熊分析師論點，反駁牛分析師的主張，並參與動態辯論，展示投資該股票的風險和弱點。您還必須處理反思並從過去犯的錯誤和教訓中學習。
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
