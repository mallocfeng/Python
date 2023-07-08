import socket

def test_tcp_server():
    # 定义服务器地址和端口号
    HOST = '127.0.0.1'
    PORT = 8888

    # 创建 TCP 客户端套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # 连接服务器
        client_socket.connect((HOST, PORT))
        print("Connected to server")

        # 发送握手请求
        client_socket.send("get <Recv_Data_Step1>".encode())

        # 接收服务器的响应
        response = client_socket.recv(1024).decode()
        print(f"Received from server: {response}")
    except ConnectionRefusedError:
        print("Server is not running or connection refused")
    finally:
        # 关闭与服务器的连接
        client_socket.close()

# 测试 TCP 服务器
test_tcp_server()

