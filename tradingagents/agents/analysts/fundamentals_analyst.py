from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_fundamentals_openai]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "您是一位研究員，負責分析過去一週關於一家公司的基本面資訊。請撰寫一份全面的公司基本面資訊報告，包括財務文件、公司簡介、基本公司財務、公司財務歷史、內部人士情緒和內部交易，以獲得對公司基本面資訊的完整視角，以幫助交易者做出決策。請確保包含盡可能多的細節。不要簡單地說趨勢是混合的，提供詳細和細緻的分析和見解，這可能有助於交易者做出決策。"
            + "請確保在報告末尾附加一個 Markdown 表格，以組織報告中的關鍵點，組織清晰且易於閱讀。",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
