from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 加载模型
model = joblib.load("recommendation_model.pkl")

# 需要的字段（顺序要和训练时一致）
required_features = [
    'family_size', 'total_income', 'requested_amount',
    'total_received_amount', 'request_receive_ratio', 'is_OKU'
]

@app.route('/')
def index():
    return "✅ Recommendation Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从请求中提取 JSON
        data = request.get_json()

        # 自动计算 ratio
        total_received = max(data['total_received_amount'], 1)  # 避免除以0
        data['request_receive_ratio'] = data['requested_amount'] / total_received

        # 创建 DataFrame
        df = pd.DataFrame([data], columns=required_features)

        # 预测
        prediction = int(model.predict(df)[0])

        return jsonify({
            "priority_class": prediction,
            "message": "✅ Prediction success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

