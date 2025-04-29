import joblib

# 设置原始 pkl 文件路径
pkl_path = "data/raw_data/first_action.pkl"  # 修改为你的路径

# 加载整个 pkl 数据（一个字典）
data = joblib.load(pkl_path)

# 提取第一个动作
# 这里假设字典的 key 顺序就是我们希望的顺序（一般取 list(data.keys())[0]）
key = list(data.keys())


# 显示提取动作的 key 和类型
print("all key：", key)

print(data['pose_aa'][0].shape)