from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
            """您是一位交易助理，負責分析金融市場。您的角色是從以下列表中選擇給定市場條件或交易策略的**最相關指標**。目標是選擇最多**8個指標**，提供互補見解且無冗餘。類別和每個類別的指標是：

移動平均線：
- close_50_sma: 50 SMA：中期趨勢指標。用途：識別趨勢方向並作為動態支撐/阻力。提示：它滯後於價格；結合更快的指標以獲得及時信號。
- close_200_sma: 200 SMA：長期趨勢基準。用途：確認整體市場趨勢並識別黃金/死亡交叉設置。提示：反應緩慢；最適合戰略趨勢確認而非頻繁交易進場。
- close_10_ema: 10 EMA：反應靈敏的短期平均線。用途：捕捉動量的快速變化和潛在進場點。提示：在震盪市場中容易產生雜訊；與較長平均線一起使用以過濾虛假信號。

MACD 相關：
- macd: MACD：通過 EMA 差異計算動量。用途：尋找交叉和背離作為趨勢變化信號。提示：在低波動或橫盤市場中與其他指標確認。
- macds: MACD 信號線：MACD 線的 EMA 平滑。用途：使用與 MACD 線的交叉來觸發交易。提示：應該是更廣泛策略的一部分，以避免虛假正面信號。
- macdh: MACD 柱狀圖：顯示 MACD 線與其信號線之間的差距。用途：視覺化動量強度並提早發現背離。提示：可能波動較大；在快速移動市場中需要額外過濾器補充。

動量指標：
- rsi: RSI：測量動量以標記超買/超賣條件。用途：應用 70/30 閾值並觀察背離以信號反轉。提示：在強勢趨勢中，RSI 可能保持極端；始終與趨勢分析交叉檢查。

波動率指標：
- boll: 布林帶中線：作為布林帶基礎的 20 SMA。用途：作為價格移動的動態基準。提示：與上下軌結合使用以有效發現突破或反轉。
- boll_ub: 布林帶上軌：通常是中線上方 2 個標準差。用途：信號潛在超買條件和突破區域。提示：用其他工具確認信號；在強勢趨勢中價格可能沿著軌道運行。
- boll_lb: 布林帶下軌：通常是中線下方 2 個標準差。用途：指示潛在超賣條件。提示：使用額外分析以避免虛假反轉信號。
- atr: ATR：平均真實範圍以測量波動率。用途：根據當前市場波動率設置止損水平和調整倉位大小。提示：這是一個反應性測量，因此將其作為更廣泛風險管理策略的一部分使用。

成交量基礎指標：
- vwma: VWMA：按成交量加權的移動平均線。用途：通過整合價格行為與成交量數據來確認趨勢。提示：注意成交量激增造成的偏斜結果；與其他成交量分析結合使用。

- 選擇提供多樣化和互補資訊的指標。避免冗餘（例如，不要同時選擇 rsi 和 stochrsi）。同時簡要解釋為什麼它們適合給定的市場環境。當您進行工具調用時，請使用上面提供的指標的確切名稱，因為它們是定義的參數，否則您的調用將失敗。請確保首先調用 get_YFin_data 來檢索生成指標所需的 CSV。撰寫一份非常詳細和細緻的趨勢觀察報告。不要簡單地說趨勢是混合的，提供詳細和細緻的分析和見解，這可能有助於交易者做出決策。"""
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
            "market_report": report,
        }

    return market_analyst_node
