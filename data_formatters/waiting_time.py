# coding=utf-8
"""针对车辆等待时间数据集的自定义格式化函数。

定义数据集特定的列定义和数据转换。
"""

import pandas as pd
import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class WaitingTimeFormatter(GenericDataFormatter):
    """定义和格式化车辆等待时间数据集。

    属性:
        column_definition: 定义实验中使用的列的输入和数据类型
        identifiers: 实验中使用的实体标识符
    """

    _column_definition = [
        ('提煤单号', DataTypes.CATEGORICAL, InputTypes.ID),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('等待时间', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('煤种编号', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('minute', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('weekday', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('is_weekend', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('当前队列长度', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('煤种平均处理时间', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('预计等待时间', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('历史平均等待时间', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
    ]

    def __init__(self):
        """初始化格式化程序"""
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def get_column_definition(self):
        """返回数据集的列定义。"""
        column_definition = [
            ('提煤单号', DataTypes.CATEGORICAL, InputTypes.ID),
            ('date', DataTypes.DATE, InputTypes.TIME),
            ('等待时间', DataTypes.REAL_VALUED, InputTypes.TARGET),
            ('煤种编号', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('minute', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('weekday', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('is_weekend', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('当前队列长度', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('煤种平均处理时间', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('预计等待时间', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('历史平均等待时间', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ]
        
        return column_definition

    def split_data(self, df):
        """将数据分割为训练集、验证集和测试集。"""
        # 按时间顺序排序
        df = df.sort_values('date')
        
        # 获取所有唯一的煤种
        coal_types = df['煤种编号'].unique()
        
        # 为每个煤种分别划分数据
        train_dfs = []
        valid_dfs = []
        test_dfs = []
        
        for coal_type in coal_types:
            coal_data = df[df['煤种编号'] == coal_type].copy()
            n = len(coal_data)
            
            # 使用固定比例划分：70% 训练，15% 验证，15% 测试
            train_end = int(n * 0.7)
            valid_end = int(n * 0.85)
            
            train_dfs.append(coal_data.iloc[:train_end])
            valid_dfs.append(coal_data.iloc[train_end:valid_end])
            test_dfs.append(coal_data.iloc[valid_end:])
        
        # 合并所有煤种的数据
        train = pd.concat(train_dfs, axis=0).sort_values('date')
        valid = pd.concat(valid_dfs, axis=0).sort_values('date')
        test = pd.concat(test_dfs, axis=0).sort_values('date')
        
        # 重置索引
        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        test = test.reset_index(drop=True)
        
        # 存储标识符列表
        id_col = None
        for col, dtype, input_type in self.get_column_definition():
            if input_type == InputTypes.ID:
                id_col = col
                break
        
        if id_col is not None:
            self.identifiers = list(df[id_col].unique())
        else:
            raise ValueError("未找到ID列")
        
        return self.transform_data(train, valid, test)

    def transform_data(self, train, valid, test):
        """转换所有数据集。
        
        Args:
            train: 训练数据
            valid: 验证数据
            test: 测试数据
            
        Returns:
            转换后的数据集
        """
        # 使用训练集来训练定标器
        self.set_scalers(train)
        
        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """设置定标器。"""
        print('设置定标器...')
        
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                      column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                          column_definitions)

        # 提取数值列
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # 提取分类列
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # 设置数值定标器
        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        
        # 设置目标定标器
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)

        # 设置分类定标器
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # 确保所有值都是字符串类型
            srs = df[col].astype(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """执行特征转换。

        包括特征工程、预处理和归一化。

        Args:
            df: 要转换的数据帧。

        Returns:
            转换后的数据帧。
        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('定标器尚未设置！')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """将预测结果转换回原始比例。

        Args:
            predictions: 模型预测结果。

        Returns:
            转换后的预测结果。
        """
        output = pd.DataFrame()
        
        for col in predictions.columns:
            if col.startswith('t+'):
                values = predictions[col].values.reshape(-1, 1)
                output[col] = self._target_scaler.inverse_transform(values).flatten()
            else:
                output[col] = predictions[col]

        return output

    def get_fixed_params(self):
        """返回固定的模型参数。"""
        return {
            'total_time_steps': 25,
            'num_encoder_steps': 24,
            'num_epochs': 1,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

    def get_default_model_params(self):
        """返回默认的优化模型参数。"""
        return {
            'dropout_rate': 0.3,
            'hidden_layer_size': 128,
            'learning_rate': 0.01,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 4,
            'stack_size': 1
        }

    def get_num_samples_for_calibration(self):
        """返回用于校准的样本数。
        
        Returns:
            tuple: (训练样本数, 验证样本数)
        """
        return 100000, 20000  # 返回元组：(训练样本数, 验证样本数)