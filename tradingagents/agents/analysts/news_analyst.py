from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_global_news_openai, toolkit.get_google_news]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            "您是一位新聞研究員，負責分析過去一週的新聞和趨勢。請撰寫一份全面的報告，涵蓋當前世界狀態，這對交易和宏觀經濟學都相關。查看 EODHD 和 finnhub 的新聞，以確保全面。不要簡單地說趨勢是混合的，提供詳細和細緻的分析和見解，這可能有助於交易者做出決策。"
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
            "news_report": report,
        }

    return news_analyst_node
