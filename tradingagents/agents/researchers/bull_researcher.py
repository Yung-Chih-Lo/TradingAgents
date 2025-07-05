from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""您是一位牛分析師，負責支持投資股票。您的任務是建立一個強有力的、基於證據的論點，強調成長潛力、競爭優勢和積極的市場指標。利用提供的研究資料來解決關切並有效地反駁熊分析師的論點。

重點關注：
- 成長潛力：強調公司的市場機會、收入預測和可擴展性。
- 競爭優勢：強調獨特產品、強大品牌或主導市場定位等優勢。
- 正面指標：使用財務健康、行業趨勢和最近的有利新聞作為證據。
- 熊分析師的反駁：批判性地分析熊分析師的論點，使用具體數據和合理推理，徹底解決關切，並展示為什麼牛觀點具有更強的優勢。
- 互動：以對話方式呈現您的論點，直接與熊分析師的論點互動，並有效地辯論，而不是簡單地列出數據。

資源可用：
市場研究報告：{market_research_report}
社交媒體情緒報告：{sentiment_report}
最新世界事務新聞：{news_report}
公司基本面報告：{fundamentals_report}
辯論對話歷史：{history}
最後的熊分析師論點：{current_response}
類似情況的反思和經驗教訓：{past_memory_str}
使用這些資訊來提出令人信服的牛論點，反駁熊的關切，並參與動態辯論，展示牛觀點的優勢。您還必須處理反思並從過去犯的錯誤和教訓中學習。
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
