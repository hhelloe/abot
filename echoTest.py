from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def receive_json():
    if request.method == 'POST':
        try:
            # 获取 JSON 数据
            json_data = request.get_json()

            # 将 JSON 转换为字符串并打印
            json_str = str(json_data)
            print("接收到的 JSON 数据:")
            print(json_str)

            return jsonify({"status": "success", "message": "数据已接收"})

        except Exception as e:
            print(f"发生错误: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 400

    elif request.method == 'GET':
        return "服务器正在运行，请使用 POST 请求发送 JSON 数据"

if __name__ == '__main__':
    print("开始监听 5708 端口的 HTTP 请求...")
    app.run(host='0.0.0.0', port=5708, debug=True)