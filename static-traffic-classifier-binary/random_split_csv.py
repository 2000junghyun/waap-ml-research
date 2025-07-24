import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_csv('ML-lab/data/merged_data.csv')

# 4:1 비율로 랜덤 분할
train_df, test_df = train_test_split(
    df,
    test_size=0.2,      # 4:1 분할
    random_state=42,    # 재현성(항상 같은 결과)
    shuffle=True        # 랜덤하게 섞어서 분할
)

# 저장
train_df.to_csv('ML-lab/data/train_data.csv', index=False)
test_df.to_csv('ML-lab/data/test_data.csv', index=False)

print("CSV split complete.")