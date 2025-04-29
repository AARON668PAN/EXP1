import os
import cv2
import time
import requests
from openai import OpenAI
import time



client = OpenAI(
    api_key="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"
)


# 用户输入
video_path = '/home/lingfan/retarget_data/data/videos/walk.mp4'  # 例如 /home/lingfan/retarget_data/your_video.mp4
frames_dir = "/home/lingfan/Pictures/4oVision/motion1"  # 你的HTTP服务器目录
ngrok_domain = 'https://90e3-77-241-76-113.ngrok-free.app'  # 例如 https://df17-xxxx.ngrok-free.app


# 抽取5帧函数
def extract_n_frames(video_path, n_frames=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // n_frames)
    print("总帧数:", total_frames)

    extracted_paths = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            save_path = os.path.join(frames_dir, f"frame_{i}.png")
            cv2.imwrite(save_path, frame)
            extracted_paths.append(save_path)
    cap.release()
    return extracted_paths

# 抽帧并保存
print("开始抽帧...")
frame_paths = extract_n_frames(video_path, n_frames=5)
print(f"抽取并保存了 {len(frame_paths)} 张帧图。")

# 生成公网图片url
image_urls = []
for path in frame_paths:
    filename = os.path.basename(path)
    public_url = f"{ngrok_domain}/{filename}"
    image_urls.append(public_url)



start_time = time.time()
vision_inputs = [
    {
        "type": "text", 
        "text": "根据以下多张图片描述人物动作。仅用简单动词或动词短语回答，不要完整句子，不描述背景环境、服装或其他细节。"
    }
]

for url in image_urls:
    vision_inputs.append({
        "type": "image_url",
        "image_url": {"url": url}
    })

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": vision_inputs
        }
    ],
    max_tokens=500,
)
end_time = time.time()

print('time cost is', start_time - end_time)

print(response.choices[0].message.content)
