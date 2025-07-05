import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        context = {
            "role": "user",
            "content": f"基於團隊分析師的全面分析，以下是為 {company_name} 量身定制的投資計劃。這個計劃結合了當前技術市場趨勢、宏觀經濟指標和社交媒體情緒的見解。使用這個計劃作為評估您下一筆交易的基礎。\n\n擬議投資計劃：{investment_plan}\n\n利用這些見解做出明智且有策略性的決定。",
        }

        messages = [
            {
                "role": "system",
                "content": f"""您是一位交易員，負責分析市場資料以做出投資決策。基於您的分析，提供具體的建議買入、賣出或持有。最後以堅定的決定結束，並在您的回應中始終結束 '最終交易提案：**買入/持有/賣出**' 以確認您的建議。不要忘記利用過去決策的教訓來學習您的錯誤。以下是一些類似情況的反思和您從中學到的教訓：{past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
