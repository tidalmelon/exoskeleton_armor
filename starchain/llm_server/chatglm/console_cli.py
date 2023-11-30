import requests
import json

headers={
    'Authorization': '',
    'Content-Type': 'application/json'
}

history = []

payload = {
    "prompt": "刘备生平",
    "temperature": 0.05,
    "history": history,
    "max_length": 10000,
    "top_p": 0.7,
}


#port = 4001
port = 4002


print('1. 打印机模式')
url = f'http://127.0.0.1:{port}/stream'
response = requests.post(url, stream=True, headers=headers, json=payload)
if response.status_code == 200:
    for line in response.iter_lines(decode_unicode=True):
        if line:
            if line.startswith("data"):
                line = line[6:]
                dic = json.loads(line)
                if 'prompt' not in dic:
                    print(dic['delta'], end="", flush=True)
                else:
                    print()
print('1. 打印机模式, 测试完成\n==============================')

#print("2. 堵塞模式")
#url = f'http://127.0.0.1:{port}/chat'
#response = requests.post(url, headers=headers, json=payload)
#if response.status_code == 200:
#    print(response.json()['response'])
#
#print("2. 堵塞模式测试完成\n==============================")

