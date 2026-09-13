import urllib.request, json

samples = {
    "1_接听": "您好，这里是智能客服，请问有什么可以帮您？",
    "2_拨号": "好的，马上帮您拨打张总的电话，请稍等。",
    "3_转接": "张三的电话占线中，需要我稍后再试一次吗？",
}
boundary = "----X"
for name, text in samples.items():
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n{text}\r\n"
            f"--{boundary}--\r\n").encode()
    req = urllib.request.Request("http://localhost:9880/tts/vits", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    data = urllib.request.urlopen(req, timeout=120).read()
    open(f"/tmp/sample_{name}.wav", "wb").write(data)
    print(f"生成: {name} ({len(data)} bytes)")
