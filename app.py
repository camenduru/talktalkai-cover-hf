# adapted for Zero GPU on Hugging Face

import spaces

import os
import glob
import json
import traceback
import logging
import gradio as gr
import numpy as np
import librosa
import torch
import asyncio
import ffmpeg
import subprocess
import sys
import io
import wave
from datetime import datetime
#from fairseq import checkpoint_utils
import urllib.request
import zipfile
import shutil
import gradio as gr
from textwrap import dedent
import pprint
import time

import re
import requests
import subprocess
from pathlib import Path
from scipy.io.wavfile import write
from scipy.io import wavfile
import soundfile as sf

from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from config import Config
config = Config()
logging.getLogger("numba").setLevel(logging.WARNING)
spaces_hf = True #os.getenv("SYSTEM") == "spaces"
force_support = True

audio_mode = []
f0method_mode = []
f0method_info = ""

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}
pattern = r'//www\.bilibili\.com/video[^"]*'

# Download models

#urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/hubert_base", "hubert_base.pt")
#urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/rmvpe", "rmvpe.pt")

# Get zip name

pattern_zip = r"/([^/]+)\.zip$"

def get_file_name(url):
  match = re.search(pattern_zip, url)
  if match:
      extracted_string = match.group(1)
      return extracted_string
  else:
      raise Exception("没有找到AI歌手模型的zip压缩包。")

# Get RVC models

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise Exception(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    # move model and index file to extraction folder
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # remove any unnecessary nested folders
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

# Get username in OpenXLab

def get_username(url):
    match_username = re.search(r'models/(.*?)/', url)
    if match_username:
        result = match_username.group(1)
        return result

def download_online_model(url, dir_name):
    if url.startswith('https://download.openxlab.org.cn/models/'):
        zip_path = get_username(url) + "-" + get_file_name(url)
    else:
        zip_path = get_file_name(url)
    if not os.path.exists(zip_path):
      try:
          zip_name = url.split('/')[-1]
          extraction_folder = os.path.join(zip_path, dir_name)
          if os.path.exists(extraction_folder):
              raise Exception(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

          if 'pixeldrain.com' in url:
              url = f'https://pixeldrain.com/api/file/{zip_name}'

          urllib.request.urlretrieve(url, zip_name)

          extract_zip(extraction_folder, zip_name)
          #return f'[√] {dir_name} Model successfully downloaded!'

      except Exception as e:
          raise Exception(str(e))

#Get bilibili BV id

def get_bilibili_video_id(url):
    match = re.search(r'/video/([a-zA-Z0-9]+)/', url)
    extracted_value = match.group(1)
    return extracted_value

# Get bilibili audio
def find_first_appearance_with_neighborhood(text, pattern):
    match = re.search(pattern, text)

    if match:
        return match.group()
    else:
        return None

def search_bilibili(keyword):
    if keyword.startswith("BV"):
        req = requests.get("https://search.bilibili.com/all?keyword={}&duration=1".format(keyword), headers=headers).text
    else:
        req = requests.get("https://search.bilibili.com/all?keyword={}&duration=1&tids=3&page=1".format(keyword), headers=headers).text

    video_link = "https:" + find_first_appearance_with_neighborhood(req, pattern)

    return video_link

# Save bilibili audio

def get_response(html_url):
  headers = {
      "referer": "https://www.bilibili.com/",
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
  }
  response = requests.get(html_url, headers=headers)
  return response

def get_video_info(html_url):
  response = get_response(html_url)
  html_data = re.findall('<script>window.__playinfo__=(.*?)</script>', response.text)[0]
  json_data = json.loads(html_data)
  if json_data['data']['dash']['audio'][0]['backupUrl']!=None:
    audio_url = json_data['data']['dash']['audio'][0]['backupUrl'][0]
  else:
    audio_url = json_data['data']['dash']['audio'][0]['baseUrl']
  return audio_url

def save_audio(title, audio_url):
  audio_content = get_response(audio_url).content
  with open(title + '.wav', mode='wb') as f:
    f.write(audio_content)
  print("音乐内容保存完成")


# Use UVR-HP5/2

urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP2.pth", "uvr5/uvr_model/UVR-HP2.pth")
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP5.pth", "uvr5/uvr_model/UVR-HP5.pth")
#urllib.request.urlretrieve("https://huggingface.co/fastrolling/uvr/resolve/main/Main_Models/5_HP-Karaoke-UVR.pth", "uvr5/uvr_model/UVR-HP5.pth")

from uvr5.vr import AudioPre
weight_uvr5_root = "uvr5/uvr_model"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

func = AudioPre
pre_fun_hp2 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP2.pth"),
  device="cuda",
  is_half=True,
)

pre_fun_hp5 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP5.pth"),
  device="cuda",
  is_half=True,
)

# Separate vocals

# GPU needed
@spaces.GPU(duration=120)
def get_vocal_gpu(audio_path, split_model, filename):
    if split_model=="UVR-HP2":
        pre_fun = pre_fun_hp2
    else:
        pre_fun = pre_fun_hp5
    return pre_fun._path_audio_(audio_path, f"./output/{split_model}/{filename}/", f"./output/{split_model}/{filename}/", "wav")

def youtube_downloader(
    video_identifier,
    filename,
    split_model,
):
    print(video_identifier)
    video_info = get_video_info(video_identifier)
    print(video_info)
    audio_content = get_response(video_info).content
    with open(filename.strip() + ".wav", mode="wb") as f:
        f.write(audio_content)
    audio_path = filename.strip() + ".wav"

      # make dir output
    os.makedirs("output", exist_ok=True)

    get_vocal_gpu(audio_path, split_model, filename)
    #pre_fun._path_audio_(audio_path, f"./output/{split_model}/{filename}/", f"./output/{split_model}/{filename}/", "wav")
    os.remove(filename.strip()+".wav")
    
    return f"./output/{split_model}/{filename}/vocal_{filename}.wav_10.wav", f"./output/{split_model}/{filename}/instrument_{filename}.wav_10.wav"

# Original code

if force_support is False or spaces_hf is True:
    if spaces_hf is True:
        audio_mode = ["Upload audio", "TTS Audio"]
    else:
        audio_mode = ["Input path", "Upload audio", "TTS Audio"]
    f0method_mode = ["pm", "harvest"]
    f0method_info = "PM is fast, Harvest is good but extremely slow, Rvmpe is alternative to harvest (might be better). (Default: PM)"
else:
    audio_mode = ["Input path", "Upload audio", "Youtube", "TTS Audio"]
    f0method_mode = ["pm", "harvest", "crepe"]
    f0method_info = "PM is fast, Harvest is good but extremely slow, Rvmpe is alternative to harvest (might be better), and Crepe effect is good but requires GPU (Default: PM)"

if os.path.isfile("rmvpe.pt"):
    f0method_mode.insert(2, "rmvpe")

def create_vc_fn(model_name, tgt_sr, net_g, vc, if_f0, version, file_index):
    def vc_fn(
        vc_audio_mode,
        vc_input,
        vc_upload,
        tts_text,
        tts_voice,
        f0_up_key,
        f0_method,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        try:
            logs = []
            print(f"Converting using {model_name}...")
            logs.append(f"Converting using {model_name}...")
            yield "\n".join(logs), None
            if vc_audio_mode == "Input path" or "Youtube" and vc_input != "":
                audio, sr = librosa.load(vc_input, sr=16000, mono=True)
            elif vc_audio_mode == "Upload audio":
                if vc_upload is None:
                    return "You need to upload an audio", None
                sampling_rate, audio = vc_upload
                duration = audio.shape[0] / sampling_rate
                if duration > 20 and spaces_hf:
                    return "Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.", None
                audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio.transpose(1, 0))
                if sampling_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            times = [0, 0, 0]
            f0_up_key = int(f0_up_key)
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                vc_input,
                times,
                f0_up_key,
                f0_method,
                file_index,
                # file_big_npy,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                f0_file=None,
            )
            info = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
            print(f"{model_name} | {info}")
            logs.append(f"Successfully Convert {model_name}\n{info}")
            yield "\n".join(logs), (tgt_sr, audio_opt)
        except Exception as err:
            info = traceback.format_exc()
            print(info)
            print(f"Error when using {model_name}.\n{str(err)}")
            yield info, None
    return vc_fn

def combine_vocal_and_inst(model_name, song_name, song_id, split_model, cover_song, vocal_volume, inst_volume):
    #samplerate, data = wavfile.read(cover_song)
    vocal_path = cover_song #f"output/{split_model}/{song_id}/vocal_{song_id}.wav_10.wav"
    output_path = song_name.strip() + "-AI-" + ''.join(os.listdir(f"{model_name}")).strip() + "翻唱版.mp3"
    inst_path = f"output/{split_model}/{song_id}/instrument_{song_id}.wav_10.wav"
    #with wave.open(vocal_path, "w") as wave_file:
        #wave_file.setnchannels(1)
        #wave_file.setsampwidth(2)
        #wave_file.setframerate(samplerate)
        #wave_file.writeframes(data.tobytes())
    command =  f'ffmpeg -y -i {inst_path} -i {vocal_path} -filter_complex [0:a]volume={inst_volume}[i];[1:a]volume={vocal_volume}[v];[i][v]amix=inputs=2:duration=longest[a] -map [a] -b:a 320k -c:a libmp3lame {output_path}'
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    print(result.stdout.decode())
    return output_path


@spaces.GPU()
def load_hubert():
    global hubert_model
    from fairseq import checkpoint_utils

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

print("0.开始加载Hubert")
load_hubert()

def rvc_models(model_name):
  global vc, net_g, index_files, tgt_sr, version
  categories = []
  models = []
  for w_root, w_dirs, _ in os.walk(f"{model_name}"):
      model_count = 1
      for sub_dir in w_dirs:
          pth_files = glob.glob(f"{model_name}/{sub_dir}/*.pth")
          index_files = glob.glob(f"{model_name}/{sub_dir}/*.index")
          if pth_files == []:
              print(f"Model [{model_count}/{len(w_dirs)}]: No Model file detected, skipping...")
              continue
          cpt = torch.load(pth_files[0], map_location="cpu")
          tgt_sr = cpt["config"][-1]
          cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
          if_f0 = cpt.get("f0", 1)
          version = cpt.get("version", "v1")
          if version == "v1":
              if if_f0 == 1:
                  net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
              else:
                  net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
              model_version = "V1"
          elif version == "v2":
              if if_f0 == 1:
                  net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
              else:
                  net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
              model_version = "V2"
          del net_g.enc_q
          print(net_g.load_state_dict(cpt["weight"], strict=False))
          net_g.eval().to(config.device)
          if config.is_half:
              net_g = net_g.half()
          else:
              net_g = net_g.float()
          vc = VC(tgt_sr, config)
          if index_files == []:
              print("Warning: No Index file detected!")
              index_info = "None"
              model_index = ""
          else:
              index_info = index_files[0]
              model_index = index_files[0]
          print(f"Model loaded [{model_count}/{len(w_dirs)}]: {index_files[0]} / {index_info} | ({model_version})")
          model_count += 1
          models.append((index_files[0][:-4], index_files[0][:-4], "", "", model_version, create_vc_fn(index_files[0], tgt_sr, net_g, vc, if_f0, version, model_index)))
  categories.append(["Models", "", models])
  return vc, net_g, index_files, tgt_sr, version


singers="您的专属AI歌手阵容:"

@spaces.GPU(duration=60)
def infer_gpu(hubert_model, net_g, audio, f0_up_key, index_file, tgt_sr, version, f0_file=None):
    return vc.pipeline(
          hubert_model,
          net_g,
          0,
          audio,
          "",
          [0, 0, 0],
          f0_up_key,
          "rmvpe",
          index_file,
          0.7,
          1,
          3,
          tgt_sr,
          0,
          0.25,
          version,
          0.33,
          f0_file=None,
    )
    
def rvc_infer_music(url, model_name, song_name, split_model, f0_up_key, vocal_volume, inst_volume):
  load_hubert()
  url = url.strip().replace(" ", "")
  model_name = model_name.strip().replace(" ", "")
  if url.startswith('https://download.openxlab.org.cn/models/'):
      zip_path = get_username(url) + "-" + get_file_name(url)
  else:
      zip_path = get_file_name(url)
  global singers
  if model_name not in singers:
    singers = singers+ '   '+ model_name
  print("1.开始下载模型")
  download_online_model(url, model_name)
  rvc_models(zip_path)
  song_name = song_name.strip().replace(" ", "")
  video_identifier = search_bilibili(song_name)
  song_id = get_bilibili_video_id(video_identifier)
  if os.path.isdir(f"./output/{split_model}/{song_id}")==True:
    print("2.直接开始推理")
    audio, sr = librosa.load(f"./output/{split_model}/{song_id}/vocal_{song_id}.wav_10.wav", sr=16000, mono=True)
    song_infer = infer_gpu(hubert_model, net_g, audio, f0_up_key, index_files[0], tgt_sr, version, f0_file=None)
  else:
    print("2.1.开始去除BGM")
    audio, sr = librosa.load(youtube_downloader(video_identifier, song_id, split_model)[0], sr=16000, mono=True)
    print("2.2.开始推理")
    song_infer = infer_gpu(hubert_model, net_g, audio, f0_up_key, index_files[0], tgt_sr, version, f0_file=None)

  sf.write(song_name.strip()+zip_path+"AI翻唱.wav", song_infer, tgt_sr)
  output_full_song = combine_vocal_and_inst(zip_path, song_name.strip(), song_id, split_model, song_name.strip()+zip_path+"AI翻唱.wav", vocal_volume, inst_volume)
  os.remove(song_name.strip()+zip_path+"AI翻唱.wav")
  return output_full_song, singers

app = gr.Blocks(theme="JohnSmith9982/small_and_pretty")
with app:
    with gr.Tab("中文版"):
      gr.Markdown("# <center>🌊💕🎶 滔滔AI，您的专属AI全明星乐团</center>")
      gr.Markdown("## <center>🌟 只需一个歌曲名，全网AI歌手任您选择！随时随地，听我想听！</center>")
      gr.Markdown("### <center>🤗 更多精彩应用，敬请关注[滔滔AI](http://www.talktalkai.com)；相关问题欢迎在我们的[B站](https://space.bilibili.com/501495851)账号交流！滔滔AI，为爱滔滔！💕</center>")
      with gr.Accordion("💡 一些AI歌手模型链接及使用说明（建议阅读）", open=False):
          _ = f""" 任何能够在线下载的zip压缩包的链接都可以哦（zip压缩包只需包括AI歌手模型的.pth和.index文件，zip压缩包的链接需要以.zip作为后缀）:
              * Taylor Swift: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/taylor.zip
              * Blackpink Lisa: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/Lisa.zip
              * AI派蒙: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/paimon.zip
              * AI孙燕姿: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/syz.zip
              * AI[一清清清](https://www.bilibili.com/video/BV1wV411u74P)（推荐使用[OpenXLab](https://openxlab.org.cn/models)存放模型zip压缩包）: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/yiqing.zip\n
              说明1：点击“一键开启AI翻唱之旅吧！”按钮即可使用！✨\n
              说明2：一般情况下，男声演唱的歌曲转换成AI女声演唱需要升调，反之则需要降调；在“歌曲人声升降调”模块可以调整\n
              说明3：对于同一个AI歌手模型或者同一首歌曲，第一次的运行时间会比较长（大约1分钟），请您耐心等待；之后的运行时间会大大缩短哦！\n
              说明4：您之前下载过的模型会在“已下载的AI歌手全明星阵容”模块出现\n
              说明5：此程序使用 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) AI歌手模型，感谢[作者](https://space.bilibili.com/5760446)的开源！RVC模型训练教程参见[视频](https://www.bilibili.com/video/BV1mX4y1C7w4)\n
              🤗 我们正在创建一个完全开源、共建共享的AI歌手模型社区，让更多的人感受到AI音乐的乐趣与魅力！请关注我们的[B站](https://space.bilibili.com/501495851)账号，了解社区的最新进展！合作联系：talktalkai.kevin@gmail.com
              """
          gr.Markdown(dedent(_))
    
      with gr.Row():
        with gr.Column():
          inp1 = gr.Textbox(label="请输入AI歌手模型链接", info="模型需要是含有.pth和.index文件的zip压缩包", lines=2, value="https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/taylor.zip", placeholder="https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/taylor.zip")
        with gr.Column():
          inp2 = gr.Textbox(label="请给您的AI歌手起一个昵称吧", info="可自定义名称，但名称中不能有特殊符号", lines=1, value="AI Taylor", placeholder="AI Taylor")
          inp3 = gr.Textbox(label="请输入您需要AI翻唱的歌曲名", info="如果您对搜索结果不满意，可在歌曲名后加上“无损”或“歌手的名字”等关键词；歌曲名中不能有特殊符号", lines=1, value="小幸运", placeholder="小幸运")
      with gr.Row():
        inp4 = gr.Dropdown(label="请选择用于分离伴奏的模型", choices=["UVR-HP2", "UVR-HP5"], value="UVR-HP5", visible=False)
        inp5 = gr.Slider(label="歌曲人声升降调", info="默认为0，+2为升高2个key，以此类推", minimum=-12, maximum=12, value=0, step=1)
        inp6 = gr.Slider(label="歌曲人声音量调节", info="默认为1，等于0时为静音", minimum=0, maximum=3, value=1, step=0.2)
        inp7 = gr.Slider(label="歌曲伴奏音量调节", info="默认为1，等于0时为静音", minimum=0, maximum=3, value=1, step=0.2)
        btn = gr.Button("一键开启AI翻唱之旅吧！💕", variant="primary")
      with gr.Row():
        output_song = gr.Audio(label="AI歌手为您倾情演绎")
        singer_list = gr.Textbox(label="已下载的AI歌手全明星阵容")
    
      btn.click(fn=rvc_infer_music, inputs=[inp1, inp2, inp3, inp4, inp5, inp6, inp7], outputs=[output_song, singer_list])
    
      gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。请自觉合规使用此程序，程序开发者不负有任何责任。</center>")
      gr.HTML('''
          <div class="footer">
                      <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                      </p>
          </div>
      ''')
    with gr.Tab("EN"):
      gr.Markdown("# <center>🌊💕🎶 TalkTalkAI - Best AI song cover generator ever</center>")
      gr.Markdown("## <center>🌟 Provide the name of a song and our application running on A100 will handle everything else!</center>")
      gr.Markdown("### <center>🤗 [TalkTalkAI](http://www.talktalkai.com/), let everyone enjoy a better life through human-centered AI💕</center>")
      with gr.Accordion("💡 Some AI singers you can try", open=False):
          _ = f""" Any Zip file that you can download online will be fine (The Zip file should contain .pth and .index files):
              * AI Taylor Swift: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/taylor.zip
              * AI Blackpink Lisa: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/Lisa.zip
              * AI Paimon: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/paimon.zip
              * AI Stefanie Sun: https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/syz.zip
              * AI[一清清清](https://www.bilibili.com/video/BV1wV411u74P): https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/yiqing.zip\n
              """
          gr.Markdown(dedent(_))
    
      with gr.Row():
        with gr.Column():
          inp1_en = gr.Textbox(label="The Zip file of an AI singer", info="The Zip file should contain .pth and .index files", lines=2, value="https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/taylor.zip", placeholder="https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/taylor.zip")
        with gr.Column():
          inp2_en = gr.Textbox(label="The name of your AI singer", lines=1, value="AI Taylor", placeholder="AI Taylor")
          inp3_en = gr.Textbox(label="The name of a song", lines=1, value="Hotel California Eagles", placeholder="Hotel California Eagles")
      with gr.Row():
        inp4_en = gr.Dropdown(label="UVR models", choices=["UVR-HP2", "UVR-HP5"], value="UVR-HP5", visible=False)
        inp5_en = gr.Slider(label="Transpose", info="0 from man to man (or woman to woman); 12 from man to woman and -12 from woman to man.", minimum=-12, maximum=12, value=0, step=1)
        inp6_en = gr.Slider(label="Vocal volume", info="Adjust vocal volume (Default: 1)", minimum=0, maximum=3, value=1, step=0.2)
        inp7_en = gr.Slider(label="Instrument volume", info="Adjust instrument volume (Default: 1)", minimum=0, maximum=3, value=1, step=0.2)
        btn_en = gr.Button("Convert💕", variant="primary")
      with gr.Row():
        output_song_en = gr.Audio(label="AI song cover")
        singer_list_en = gr.Textbox(label="The AI singers you have")
    
      btn_en.click(fn=rvc_infer_music, inputs=[inp1_en, inp2_en, inp3_en, inp4_en, inp5_en, inp6_en, inp7_en], outputs=[output_song_en, singer_list_en])
    

      gr.HTML('''
          <div class="footer">
                      <p>🤗 - Stay tuned! The best is yet to come.
                      </p>
                      <p>📧 - Contact us: talktalkai.kevin@gmail.com
                      </p>
          </div>
      ''')    

app.queue(max_size=40, api_open=False)
app.launch(max_threads=400, show_error=True)