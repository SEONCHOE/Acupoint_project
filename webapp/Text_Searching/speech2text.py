import os
import requests


class csr:
    def __init__(self, path):
        self.path = path

    def convert(self):
        client_id = os.environ.get("NCP_CLOVA_CLIENT_ID")
        client_secret = os.environ.get("NCP_CLOVA_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "NCP_CLOVA_CLIENT_ID / NCP_CLOVA_CLIENT_SECRET 환경변수가 설정되지 않았습니다. "
                ".env 파일을 확인하세요."
            )

        lang = "Kor"  # 언어 코드 ( Kor, Jpn, Eng, Chn )
        url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/octet-stream",
        }
        with open(self.path, 'rb') as data:
            response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            # print(response.text)
            return response.text.split(':')[1].strip("}")[1:-1]
        else:
            print("Error : " + response.text)

# model = csr('./voice.mp3')
# print(model.convert())
