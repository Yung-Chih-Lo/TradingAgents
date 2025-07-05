from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_stock_news_openai]
        else:
            tools = [
                toolkit.get_reddit_stock_info,
            ]

        system_message = (
            "您是一位社交媒體和公司特定新聞研究員/分析師，負責分析社交媒體帖子、最近的公司新聞和公共情緒，對特定公司過去一週的分析。您將獲得一家公司的名稱，您的目標是撰寫一份詳盡的長篇報告，詳細說明您的分析、見解和對交易者和投資者對這家公司的當前狀態的影響，在查看社交媒體和人們對該公司的看法後，分析人們每天對該公司的感受的情感數據，並查看最近的公司新聞。盡可能從社交媒體到情感到新聞的所有來源。不要簡單地說趨勢是混合的，提供詳細和細緻的分析和見解，這可能有助於交易者做出決策。"
            + """ 請確保在報告末尾附加一個 Markdown 表格，以組織報告中的關鍵點，組織清晰且易於閱讀。"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位有用的 AI 助手，與其他助手合作。"
                    " 使用提供的工具來推進回答問題。"
                    " 如果您無法完全回答，那也沒關係；另一個使用不同工具的助手將在您停止的地方幫助您。執行您可以做的來推進。"
                    " 如果您或任何其他助手有最終交易提案：**BUY/HOLD/SELL** 或可交付的，"
                    " 在您的回應中添加最終交易提案：**BUY/HOLD/SELL** 以便團隊知道停止。"
                    " 您有權訪問以下工具：{tool_names}。\n{system_message}"
                    " 供您參考，當前日期是 {current_date}。我們想要查看的公司是 {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
