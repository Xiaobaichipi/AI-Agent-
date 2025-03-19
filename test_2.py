import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from statsmodels.tsa.arima.model import ARIMA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

# 注册宋体字体
font_path = r'C:\\Windows\\Fonts\\simsun.ttc'  # 宋体字体路径
if os.path.exists(font_path):
    pdfmetrics.registerFont(TTFont('simsun', font_path))
else:
    raise FileNotFoundError(f"字体文件 {font_path} 不存在，请检查路径是否正确")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class StockAnalysisAndForecastToolchain:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.forecast_results = None

    def load_and_preprocess_data(self):
        # 加载数据
        self.data = pd.read_csv(self.data_path, parse_dates=['date'])
        self.data.set_index('date', inplace=True)
        # 数据清洗
        self.data.fillna(method='ffill', inplace=True)
        return "数据加载并预处理完成"

    def extract_features(self):
        # 特征提取
        self.data['rolling_mean'] = self.data['value'].rolling(window=7).mean()
        self.data['expanding_mean'] = self.data['value'].expanding().mean()
        return "特征提取完成"

    def generate_analysis_report(self):
        # 生成分析报告
        report = f"""
        # 时序数据分析报告

        ## 数据概述
        - 数据形状：{self.data.shape}
        - 数据时间范围：{self.data.index.min()} 至 {self.data.index.max()}

        ## 数据统计
        - 平均值：{self.data['value'].mean():.2f}
        - 中位数：{self.data['value'].median():.2f}
        - 标准差：{self.data['value'].std():.2f}

        ## 特征分析
        - 7日滚动平均：{float(self.data['rolling_mean'].tail(1).values[0]):.2f}
        - 累积平均：{float(self.data['expanding_mean'].tail(1).values[0]):.2f}
        """
        return report

    def plot_stock_data(self, save_path=None):
        # 绘制图表
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['value'], label='收盘价')
        ax.set_title('股票数据可视化')
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return fig

    def create_pdf_report(self, analysis_text, plot_path, forecast_plot_path, output_path):
        # 创建 PDF 文件
        doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=72, rightMargin=72, topMargin=72,
                                bottomMargin=72)
        story = []

        # 添加标题
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='ChineseTitle', parent=styles['Title'], fontName='simsun', fontSize=18))
        styles.add(ParagraphStyle(name='ChineseNormal', parent=styles['Normal'], fontName='simsun', fontSize=12))
        title = Paragraph("股票数据分析报告", styles['ChineseTitle'])
        story.append(title)

        # 添加基本统计
        basic_stats = [
            f"- 平均收盘价: {self.data['value'].mean():.2f}",
            f"- 中位数收盘价: {self.data['value'].median():.2f}",
            f"- 标准差: {self.data['value'].std():.2f}",
            f"- 最高收盘价: {self.data['value'].max():.2f}",
            f"- 最低收盘价: {self.data['value'].min():.2f}",
        ]
        basic_stats_para = Paragraph("## 基本统计<br/>" + "<br/>".join(basic_stats), styles['ChineseNormal'])
        story.append(basic_stats_para)

        # 添加趋势分析
        trend_analysis = [
            f"- 总体涨跌幅: {(self.data['value'].iloc[-1] - self.data['value'].iloc[0]) / self.data['value'].iloc[0]:.2%}",
            f"- 波动率: {self.data['value'].pct_change().std():.2%}",
        ]
        trend_analysis_para = Paragraph("## 趋势分析<br/>" + "<br/>".join(trend_analysis), styles['ChineseNormal'])
        story.append(trend_analysis_para)

        # 添加图表路径
        plot_path_para = Paragraph(f"## 图表<br/>- 图表路径: {plot_path}", styles['ChineseNormal'])
        story.append(plot_path_para)

        # 添加分析文本
        analysis_para = Paragraph("## 详细分析<br/>" + analysis_text.replace('\n', '<br/>'), styles['ChineseNormal'])
        story.append(analysis_para)

        # 插入图片
        img = Image(plot_path, width=400, height=200)
        story.append(img)

        # 添加预测图表路径
        forecast_plot_path_para = Paragraph(f"## 预测图表<br/>- 预测图表路径: {forecast_plot_path}",
                                            styles['ChineseNormal'])
        story.append(forecast_plot_path_para)

        # 插入预测图片
        forecast_img = Image(forecast_plot_path, width=400, height=200)
        story.append(forecast_img)

        # 构建 PDF
        doc.build(story)
        return "PDF报告生成完成"

    def perform_time_series_forecast(self, periods=30):
        # 时序预测
        model = ARIMA(self.data['value'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        self.forecast_results = forecast
        return forecast

    def plot_forecast(self, save_path=None):
        # 绘制预测图表
        if self.forecast_results is None:
            raise ValueError("请先进行时序预测")

        fig, ax = plt.subplots(figsize=(12, 6))

        # 获取最后 100 个数据点的值作为 y 轴
        y = self.data['value'][-100:]

        # 获取 y 的索引作为 x 轴
        x = y.index

        # 绘制历史数据
        ax.plot(x, y, label='历史数据')

        # 生成预测日期范围
        forecast_dates = pd.date_range(start=x[-1], periods=len(self.forecast_results), freq='D')

        # 绘制预测数据
        ax.plot(forecast_dates, self.forecast_results, label='预测数据', color='red')

        # 设置标题和标签
        ax.set_title('股票价格预测')
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.legend()

        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        return fig


# 使用示例
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    class DeepseekChat(ChatOpenAI):
        @property
        def _llm_type(self) -> str:
            return "deepseek-chat"


    llm = DeepseekChat(
        model_name="deepseek-chat",
        openai_api_key=openai_api_key,
        openai_api_base="https://api.deepseek.com"
    )
    # 初始化工具链
    toolchain = StockAnalysisAndForecastToolchain("Y603005.csv")

    # 数据加载与预处理
    toolchain.load_and_preprocess_data()

    # 特征提取
    toolchain.extract_features()

    # 生成分析报告
    analysis_report = toolchain.generate_analysis_report()

    # 绘制图表并保存
    plot_path = "stock_plot.png"
    toolchain.plot_stock_data(plot_path)

    # 进行时序预测
    toolchain.perform_time_series_forecast(periods=30)

    # 绘制预测图表并保存
    forecast_plot_path = "stock_forecast_plot.png"
    toolchain.plot_forecast(forecast_plot_path)

    # 定义分析提示模板
    analysis_prompt = ChatPromptTemplate.from_template("""
        你现在是一位数据科学专家。请根据以下数据进行分析，并撰写一份关于"{topic}"的数据分析文章。

        数据：
        {report}

        请确保内容专业、逻辑清晰，对每一项数据进行分析，来反映这支股票的变化情况，并使用吸引人的语言。每段字数为500字。
        """)

    # 定义分析链
    analysis_chain = (
            RunnablePassthrough.assign(
                topic=lambda x: x["topic"],
                report=lambda x: x["report"]
            )
            | analysis_prompt
            | llm
            | {"analysis": lambda x: x.content}
    )

    result = analysis_chain.invoke({"topic": "股票的走势与好坏", "report": analysis_report})

    # 创建PDF报告
    toolchain.create_pdf_report(result["analysis"], plot_path, forecast_plot_path,
                                "stock_analysis_and_forecast_report.pdf")

    print("分析完成，PDF报告已生成")