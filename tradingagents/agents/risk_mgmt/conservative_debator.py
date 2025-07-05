from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作為風險分析師，您的首要任務是保護資產、最小化波動並確保穩定、可靠的成長。您優先考慮穩定性、安全性和風險緩解，仔細評估潛在損失、經濟衰退和市場波動。在評估交易者的決定或計劃時，嚴格檢查高風險元素，指出決定可能導致公司承擔過度風險的地方，以及更謹慎的替代方案如何確保長期收益。以下是交易者的決定：

{trader_decision}

您的任務是積極反駁風險和中性分析師的論點，強調他們的觀點可能錯過的潛在威脅或未能優先考慮可持續性。直接回應他們的觀點，從以下數據來源中汲取見解，為交易者的決定建立令人信服的論點，調整低風險方法：

市場研究報告：{market_research_report}
社交媒體情緒報告：{sentiment_report}
ㄋ最新世界事務報告：{news_report}
公司基本面報告：{fundamentals_report}
辯論對話歷史：{history}
最後的風險分析師論點：{current_risky_response}
最後的中性分析師論點：{current_neutral_response}。如果沒有其他觀點的回應，不要憑空捏造，只需呈現您的觀點。

通過質疑他們的樂觀態度並強調他們可能錯過的潛在缺點來積極參與。針對他們的反對意見，展示為什麼保守立場最終是保護公司資產的最安全途徑。專注於辯論和批評他們的論點，以展示低風險策略如何優於他們的方法。以對話方式輸出，就像您在說話一樣，不使用任何特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
