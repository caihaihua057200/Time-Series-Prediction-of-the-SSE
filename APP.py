import akshare as ak
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
# 获取当前时间
now = datetime.now()
formatted_time = now.strftime('%Y%m%d')
index_code = "sh000001"  # 上证指数代码
df = ak.stock_zh_index_daily_em( symbol=index_code)
N=60
DF = df.tail(N)
# print(DF)
# 判断是否存在 NaN 值
if df.isnull().values.any():
    # 输出包含 NaN 的行
    nan_rows = df[df.isnull().any(axis=1)]
    print("包含 NaN 值的行如下：")
    print(nan_rows)
else:
    print("DataFrame 中没有 NaN 值。")
df['date'] = pd.to_datetime(df['date'])
# 将 'date' 列设置为索引
df.set_index('date', inplace=True)
df["date"] = df.index.day
# 需要排名的列
columns_to_rank = ['open', 'close', 'high', 'low', 'volume', 'amount']
# 按月分组，计算每组的最大值和最小值
monthly_max = df.groupby(df.index.to_period('M'))[columns_to_rank].transform('max')
monthly_min = df.groupby(df.index.to_period('M'))[columns_to_rank].transform('min')

# 按周分组，计算每组的最大值和最小值
weekly_max = df.groupby(df.index.to_period('W'))[columns_to_rank].transform('max')
weekly_min = df.groupby(df.index.to_period('W'))[columns_to_rank].transform('min')

# 将结果添加到 df 中，形成新的列
for col in columns_to_rank:
    df[f'monthly_max_{col}'] = monthly_max[col]
    df[f'monthly_min_{col}'] = monthly_min[col]
    df[f'weekly_max_{col}'] = weekly_max[col]
    df[f'weekly_min_{col}'] = weekly_min[col]

# 按月分组并对每列进行排名
for col in columns_to_rank:
    df[f'monthly_rank_{col}'] = df.groupby(df.index.to_period('M'))[col].rank()
# 按周分组并对每列进行排名
for col in columns_to_rank:
    df[f'weekly_rank_{col}'] = df.groupby(df.index.to_period('W'))[col].rank()
df['W'] = df.index.weekday
df['M'] = df.index.weekday
df['Y'] = df.index.year
# 提取时间索引的月份信息，并添加到训练数据中，创建 "month" 列
df["month"] = df.index.month
# 提取时间索引的年份信息，并添加到训练数据中，创建 "year" 列
df["year"] = df.index.year
# 提取时间索引的星期信息，并添加到训练数据中，创建 "weekday" 列
df["weekday"] = df.index.weekday
# 根据月份信息，判断是否为季报期，创建布尔型 "Quarterly_Report" 列
df["Quarterly_Report"] = df.index.month.isin([1, 4, 8, 10])
# 根据月份信息，判断是否为年报期，创建布尔型 "Quarterly_Report" 列
df["Annual_Report"] = df.index.month.isin([1, 2, 3, 4])
# 提取时间索引的黑色星期四信息，并添加到训练数据中，创建 "quarter" 列
df["Black_Thursday"] = df.index.weekday.isin([4])
# 提取时间索引的季度信息，并添加到训练数据中，创建 "quarter" 列
date_diff = df.index.to_series().diff().dt.days
# 找出长节假日的
df['holidays'] = date_diff > 3
# 将'节假日'列向后移动一行，使断层前的那一行标记为True
df['holidays'] = df['holidays'].shift(-1).fillna(False)
df["quarter"] = df.index.quarter
df = pd.get_dummies(
    data=df,        # 需要进行独热编码的 DataFrame
    columns=["date", "month", "year", "weekday"],  # 需要独热编码的列
    drop_first=True         # 删除第一列以避免多重共线性
)
def cal_range(x):
    """
    计算极差（最大值和最小值之差）。

    参数：doctest
    x (pd.Series): 输入的时间序列数据。

    返回：
    float: 极差值。

    示例：
    # >>> import pandas as pd
    # >>> x = pd.Series([1, 2, 3, 4, 5])
    # >>> cal_range(x)
    4
    """
    return x.max() - x.min()
def increase_num(x):
    """
    计算序列中发生增长的次数。

    参数：
    x (pd.Series): 输入的时间序列数据。

    返回：
    int: 序列中增长的次数。

    示例：
    # >>> x = pd.Series([1, 2, 3, 2, 4])
    # >>> increase_num(x)
    3
    """
    return (x.diff() > 0).sum()
def decrease_num(x):
    """
    计算序列中发生下降的次数。

    参数：
    x (pd.Series): 输入的时间序列数据。

    返回：
    int: 序列中下降的次数。

    示例：
    # >>> x = pd.Series([1, 2, 1, 3, 2])
    # >>> decrease_num(x)
    2
    """
    return (x.diff() < 0).sum()
def increase_mean(x):
    """
    计算序列中上升部分的均值。

    参数：
    x (pd.Series): 输入的时间序列数据。

    返回：
    float: 序列中上升部分的均值。

    示例：
    # >>> x = pd.Series([1, 2, 3, 2, 4])
    # >>> diff = x.diff()
    # >>> diff
    0    NaN
    1    1.0
    2    1.0
    3   -1.0
    4    2.0
    dtype: float64
    # >>> increase_mean(x)
    1.33
    """
    diff = x.diff()
    return diff[diff > 0].mean()
def decrease_mean(x):
    """
    计算序列中下降的均值（取绝对值）。

    参数：
    x (pd.Series): 输入的时间序列数据。

    返回：
    float: 序列中下降的均值（绝对值）。

    示例：
    # >>> import pandas as pd
    # >>> x = pd.Series([4, 3, 5, 2, 6])
    # >>> decrease_mean(x)
    2.0
    """
    diff = x.diff()
    return diff[diff < 0].abs().mean()
def increase_std(x):
    """
    计算序列中上升部分的标准差。

    参数：
    x (pd.Series): 输入的时间序列数据。

    返回：
    float: 序列中上升部分的标准差。

    示例：
    # >>> import pandas as pd
    # >>> x = pd.Series([1, 2, 3, 2, 4])
    # >>> increase_std(x)
    0.5773502691896257
    """
    diff = x.diff()
    return diff[diff > 0].std()
def decrease_std(x):
    """
    计算序列中下降部分的标准差。

    参数：
    x (pd.Series): 输入的时间序列数据。

    返回：
    float: 序列中下降部分的标准差。

    示例：
    # >>> import pandas as pd
    # >>> x = pd.Series([4, 3, 5, 2, 6])
    # >>> decrease_std(x)
    1.4142135623730951
    """
    diff = x.diff()
    return diff[diff < 0].std()
# 定义滚动窗口大小的列表
window_sizes = [30, 60, 90]
# 遍历每个窗口大小
with tqdm(window_sizes) as pbar:
    for window_size in pbar:
        # 定义要应用的聚合函数列表
        functions = ["mean", "std", "min", "max", cal_range, increase_num,
                     decrease_num, increase_mean, decrease_mean, increase_std, decrease_std]

        # 遍历每个聚合函数
        for func in functions:
            # 获取函数名称，如果是字符串则直接使用，否则使用函数的 __name__ 属性
            func_name = func if type(func) == str else func.__name__

            # 生成新列名，格式为 demand_rolling_{window_size}_{func_name}
            column_name = f"demand_rolling_{window_size}_{func_name}"

            # 计算滚动窗口的聚合值，并将结果添加到 train_data 中
            df[column_name] = df["close"].rolling(
                window=window_size,        # 滚动窗口大小
                min_periods=window_size//2,  # 最小观测值数
                closed="left"         # 滚动窗口在左侧闭合
            ).agg(func)              # 应用聚合函数
            pbar.set_postfix({"window_size": window_size, "func": func_name})
df["close_shift_1"] = df["close"].shift(1)
df["close_diff_1"] = df["close"].diff(1)
df["close_pct_1"] = df["close"].pct_change(1)
df = df.fillna(0)

window_size = 30
forecast_horizon = 5
# 初始化特征和目标列表
X = []
y = []

# 遍历数据集生成特征和目标
for i in range(len(df) - window_size - forecast_horizon + 1):
    # 提取过去 30 行数据作为特征
    X.append(df.iloc[i:i + window_size].values)

    # 提取未来 5 行的 close 列数据作为目标
    y.append(df['close'].iloc[i + window_size:i + window_size + forecast_horizon].values)

# 将特征和目标列表转换为 numpy 数组
X = np.array(X)
y = np.array(y)

# 初始化模型
lgb_model = LGBMRegressor(num_leaves=2 ** 5 - 1, n_estimators=300, verbose=-1)
linear_model = LinearRegression()

# 将训练数据展平

X_flattened = X.reshape(X.shape[0], -1)
print(X_flattened.shape)
# 训练模型
predictions_lgb = []
predictions_linear = []

for i in range(forecast_horizon):
    # 提取当前列的目标数据
    y_train_col = y[:, i]

    # 训练模型
    lgb_model.fit(X_flattened, y_train_col)
    linear_model.fit(X_flattened, y_train_col)

    # 使用最后 30 行数据进行预测
    last_30_X_flattened = X_flattened[-1]
    # 进行预测
    pred_lgb = lgb_model.predict(last_30_X_flattened[np.newaxis, :])
    pred_linear = linear_model.predict(last_30_X_flattened[np.newaxis, :])

    # 存储预测结果
    predictions_lgb.append(pred_lgb)
    predictions_linear.append(pred_linear)

# 计算每对数组的均值
mean_values = [np.mean([a1[0], a2[0]]) for a1, a2 in zip(predictions_lgb, predictions_linear)]

# 将均值列表拼接到 df 后面
mean_df = pd.DataFrame({'close': mean_values})

DF = DF.append(mean_df, ignore_index=True)
# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(DF['close'], label='Actual Values')
# Draw a vertical line to indicate the start of the predicted data
split_index = len(DF) - 6
plt.axvline(x=split_index, color='r', linestyle='--', label='Prediction Start')
# Add legend and labels
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual Values vs. Predicted Values')
plt.savefig('plot.png', dpi=300)  # You can change 'plot.png' to your desired file name and format

plt.show()