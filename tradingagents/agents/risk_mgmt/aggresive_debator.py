import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作為風險分析師，您的角色是積極支持高回報、高風險的機會，強調大膽的策略和競爭優勢。在評估交易者的決定或計劃時，專注於潛在的上升空間、成長潛力和創新效益—即使這些伴隨著較高的風險。使用提供的市場數據和情緒分析來加強您的論點並挑戰對立觀點。具體來說，直接回應保守和中性分析師提出的每一點，用數據驅動的反駁和有說服力的推理來反擊。強調他們的謹慎可能錯過關鍵機會的地方，或者他們的假設可能過於保守的地方。以下是交易者的決定：

{trader_decision}

您的任務是通過質疑和批評保守和中性觀點來為交易者的決定創造令人信服的論點，以展示為什麼您的風險回報觀點提供了最佳的前進道路。將以下來源的見解納入您的論點中：

市場研究報告：{market_research_report}
社交媒體情緒報告：{sentiment_report}
最新世界事務報告：{news_report}
公司基本面報告：{fundamentals_report}
辯論對話歷史：{history}
最後的保守分析師論點：{current_safe_response}
最後的中性分析師論點：{current_neutral_response}。如果沒有其他觀點的回應，不要憑空捏造，只需呈現您的觀點。

積極參與，針對任何具體的關切提出反駁，反駁邏輯中的弱點，並斷言風險承擔的利益，以超越市場常態。保持對辯論和說服的關注，而不是僅僅呈現數據。挑戰每個反對意見，強調為什麼高風險方法是最優的。以對話方式輸出，就像您在說話一樣，不使用任何特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
