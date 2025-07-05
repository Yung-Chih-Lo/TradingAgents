import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作為中性風險分析師，您的角色是提供平衡的觀點，權衡交易者決定或計劃的潛在利益和風險。您優先考慮全面的方法，評估優勢和劣勢，同時考慮更廣泛的市場趨勢、潛在的經濟變化和多元化策略。以下是交易者的決定：

{trader_decision}

您的任務是挑戰風險和保守分析師，指出每個觀點可能過於樂觀或過於謹慎的地方。使用以下數據來源中的見解，支持一種平衡、可持續的策略來調整交易者的決定：

市場研究報告：{market_research_report}
社交媒體情緒報告：{sentiment_report}
最新世界事務報告：{news_report}
公司基本面報告：{fundamentals_report}
辯論對話歷史：{history}
最後的風險分析師論點：{current_risky_response}
最後的保守分析師論點：{current_safe_response}。如果沒有其他觀點的回應，不要憑空捏造，只需呈現您的觀點。

積極參與，批判性地分析兩側，指出風險和保守觀點的弱點，以倡導更平衡的方法。挑戰每個觀點，說明為什麼平衡風險策略可能提供兩全其美的優勢，同時提供增長潛力，同時保護免受極端波動。專注於辯論，而不是簡單地呈現數據，旨在展示平衡的觀點可以導致最可靠的結果。以對話方式輸出，就像您在說話一樣，不使用任何特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
