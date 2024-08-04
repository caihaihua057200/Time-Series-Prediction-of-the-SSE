# 股票市场预测
本项目利用上证指数（sh000001）的历史数据来预测未来趋势。通过结合 LGBMRegressor 和 LinearRegression 模型，对未来的股价进行预测，并可视化结果。

# 依赖
- Python 3.x
- akshare
- pandas
- numpy
- scikit-learn
- lightgbm
- tqdm
- matplotlib
# 你可以使用 pip 安装所需的库：
  ```bash
  pip install akshare pandas numpy scikit-learn lightgbm tqdm matplotlib
  ```
# 使用方法
  ```bash
  python APP.py
  ```
# 数据获取：
使用 akshare 获取上证指数的日常数据。
# 数据处理：
- 基于时间索引和假期信息创建特征。
- 计算滚动窗口统计量，如均值、标准差、最小值、最大值以及自定义聚合函数。
- 特征工程：构建从历史数据中提取的特征，用于预测。
- 模型训练：使用历史数据训练 LGBMRegressor 和 LinearRegression 模型，以预测未来价格。
- 预测：结合两个模型的预测结果生成最终预测值。
- 可视化：绘制实际值与预测值的图，并将图像保存为 plot.png。
# 示例输出
脚本将生成一个实际值与预测值的图，并将其保存为 plot.png。
# 贡献
如果您有建议或改进，请随时提出问题或提交拉取请求！

# 许可证
请参见 LICENSE 文件。
