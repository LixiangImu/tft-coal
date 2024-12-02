import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class CoalDataPreprocessor:
    """煤炭运输等待时间数据预处理器"""
    
    def __init__(self, data_path):
        """初始化预处理器
        
        Args:
            data_path: 原始数据CSV文件路径
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """加载数据并转换时间格式"""
        print("加载数据...")
        # 读取CSV文件，确保正确编码
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        
        # 转换所有时间列为datetime格式
        time_columns = ['打卡时间', '入口操作时间', '回皮操作时间', '过重操作时间', '出口操作时间']
        date_columns = ['打卡日期', '入口操作日期', '回皮操作日期', '过重操作日期', '出口操作日期']
        
        # 合并日期和时间列
        for time_col, date_col in zip(time_columns, date_columns):
            col_name = time_col.replace('时间', '完整时间')
            self.df[col_name] = pd.to_datetime(
                self.df[date_col] + ' ' + self.df[time_col],
                format='%Y/%m/%d %H:%M:%S',
                errors='coerce'
            )
            
        return self
    
    def calculate_wait_time(self):
        """计算等待时间（分钟）"""
        print("计算等待时间...")
        # 计算从打卡到入口操作的等待时间
        self.df['等待时间'] = (
            self.df['入口操作完整时间'] - self.df['打卡完整时间']
        ).dt.total_seconds() / 60
        
        # 移除负值和异常值
        self.df = self.df[
            (self.df['等待时间'] >= 0) & 
            (self.df['等待时间'] <= 24 * 60)  # 最大等待24小时
        ]
        
        return self
    
    def calculate_queue_length(self):
        """计算每个煤种的实时队列长度"""
        print("计算各煤种实时队列长度...")
        
        # 按煤种分组处理
        queue_lengths = []
        for coal_type in self.df['煤种编号'].unique():
            print(f"处理煤种: {coal_type}")
            
            # 获取该煤种的所有记录
            coal_data = self.df[self.df['煤种编号'] == coal_type].copy()
            coal_data = coal_data.sort_values('打卡完整时间')
            
            # 计算每个时刻的队列长度
            for idx, row in coal_data.iterrows():
                current_time = row['打卡完整时间']
                # 计算当前时刻正在等待的车辆数量
                queue_length = len(coal_data[
                    (coal_data['打卡完整时间'] <= current_time) & 
                    (coal_data['入口操作完整时间'] > current_time)
                ])
                queue_lengths.append({
                    '提煤单号': row['提煤单号'],
                    '当前队列长度': queue_length
                })
        
        # 合并队列长度数据
        queue_df = pd.DataFrame(queue_lengths)
        self.df = self.df.merge(queue_df, on='提煤单号', how='left')
        
        print(queue_df.head())
        
        return self
    
    def add_time_features(self):
        """添加时间相关特征"""
        print("添加时间特征...")
        # 基础时间特征
        self.df['hour'] = self.df['打卡完整时间'].dt.hour
        self.df['minute'] = self.df['打卡完整时间'].dt.minute
        self.df['weekday'] = self.df['打卡完整时间'].dt.dayofweek
        self.df['is_weekend'] = self.df['weekday'].isin([5, 6]).astype(int)
        
        # 添加每个煤种的历史统计特征
        self.df = self.df.sort_values('打卡完整时间')
        for coal_type in self.df['煤种编号'].unique():
            print(f"处理煤种 {coal_type} 的历史特征...")
            mask = self.df['煤种编号'] == coal_type
            coal_data = self.df[mask].copy()
            
            # 设置时间索引用于滚动计算
            coal_data.set_index('打卡完整时间', inplace=True)
            
            # 计算移动平均等待时间（前1小时）
            rolling_mean = coal_data['等待时间'].rolling('1H', min_periods=1).mean()
            self.df.loc[mask, '历史平均等待时间'] = rolling_mean.values
            
            # 计算该煤种的整体平均处理时间
            self.df.loc[mask, '煤种平均处理时间'] = coal_data['等待时间'].mean()
            
            # 重置索引
            coal_data.reset_index(inplace=True)
        
        # 计算预计等待时间
        self.df['预计等待时间'] = self.df['当前队列长度'] * self.df['煤种平均处理时间']
        
        return self
    
    def prepare_final_dataset(self):
        """准备最终数据集"""
        print("准备最终数据集...")
        
        # 选择最终特征
        final_columns = [
            '提煤单号',          # ID
            '打卡完整时间',       # 时间戳
            '等待时间',          # 目标变量
            '煤种编号',          # 分类特征
            'hour',            # 时间特征
            'minute',
            'weekday',
            'is_weekend',
            '当前队列长度',      # 队列特征
            '煤种平均处理时间',   # 统计特征
            '预计等待时间',
            '历史平均等待时间'
        ]
        
        final_df = self.df[final_columns].copy()
        
        # 重命名时间列以符合模型要求
        final_df = final_df.rename(columns={'打卡完整时间': 'date'})
        
        # 按时间排序
        final_df = final_df.sort_values(['煤种编号', 'date'])
        
        # 保存处理后的数据
        output_path = 'data/coal/processed_coal_data.csv'
        final_df.to_csv(output_path, index=False)
        print(f"数据已保存至: {output_path}")
        
        return final_df

def main():
    """主函数"""
    print("="*50)
    print("开始数据预处理")
    print("="*50)
    
    # 初始化预处理器
    preprocessor = CoalDataPreprocessor('data/coal/coaltime.csv')
    
    # 执行预处理流程，调整方法调用顺序
    processed_data = (preprocessor
        .load_data()
        .calculate_wait_time()
        .calculate_queue_length()  # 先计算队列长度
        .add_time_features()      # 再添加时间特征
        .prepare_final_dataset()
    )
    
    print("\n" + "="*50)
    print("数据预处理完成！")
    print("="*50)
    
    print("\n数据预览:")
    print(processed_data.head())
    
    print("\n特征统计:")
    print(processed_data.describe())
    
    # 打印每个煤种的基本信息
    print("\n各煤种基本信息:")
    for coal_type in processed_data['煤种编号'].unique():
        coal_data = processed_data[processed_data['煤种编号'] == coal_type]
        print(f"\n煤种 {coal_type}:")
        print(f"  - 总车次: {len(coal_data):,} 辆")
        print(f"  - 平均等待时间: {coal_data['等待时间'].mean():.1f} 分钟")
        print(f"  - 最大队列长度: {coal_data['当前队列长度'].max()} 辆")
        print(f"  - 平均处理时间: {coal_data['煤种平均处理时间'].mean():.1f} 分钟")

if __name__ == "__main__":
    main()