import os
import sys
import torch
import numpy as np
import datetime
import soundfile as sf
from pathlib import Path

# 设置G2PW模型路径和环境变量
g2pw_zip_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                               'GPT_SoVITS', 'text', 'G2PWModel_1.1.zip'))
g2pw_model_dir = os.path.join(os.path.dirname(__file__), 
                             'GPT_SoVITS', 'text', 'G2PWModel')

# 创建环境变量，让GPT-SoVITS使用正确的路径
os.environ['G2PW_MODEL_PATH'] = g2pw_zip_path
os.environ['G2PW_MODEL_DIR'] = g2pw_model_dir

# 打印路径确认
print(f"设置G2PW模型ZIP路径: {g2pw_zip_path}")
print(f"设置G2PW模型目录: {g2pw_model_dir}")

# 添加GPT-SoVITS目录到Python路径
gpt_sovits_path = os.path.join(os.path.dirname(__file__),'GPT_SoVITS')
if gpt_sovits_path not in sys.path:
    sys.path.append(gpt_sovits_path)

# 设置G2PW模型路径环境变量
g2pw_model_dir = os.path.join(os.path.dirname(__file__),'GPT_SoVITS', 'text')
os.makedirs(g2pw_model_dir, exist_ok=True)
g2pw_model_path = os.path.join(g2pw_model_dir, 'G2PWModel_1.1.zip')
os.environ['G2PW_MODEL_DIR'] = g2pw_model_dir
print(f"G2PW模型目录: {g2pw_model_dir}")

# 修改导入语句，使用正确的函数名
from inference_webui import get_tts_wav, change_sovits_weights as load_sovits_model, change_gpt_weights as load_gpt_model

# 全局变量，保存已加载的模型
_sovits_model = None
_gpt_model = None
_hps = None
_dict_language = None

def init_models():
    """初始化并加载模型，只需执行一次"""
    global _sovits_model, _gpt_model, _hps, _dict_language
    
    if _sovits_model is not None and _gpt_model is not None:
        return  # 模型已加载，无需重复加载
    
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gpt_path = os.path.join(base_dir, "GPT_weights_v2", "白厄_extend-e20.ckpt")
    sovits_path = os.path.join(base_dir, "SoVITS_weights_v2", "白厄_extend_e12_s2664.pth")
    
    # 检查文件是否存在
    print(f"检查sovits模型文件: {os.path.exists(sovits_path)}")
    print(f"检查gpt模型文件: {os.path.exists(gpt_path)}")
    
    # 如果文件不存在，尝试查找其他可能的路径
    if not os.path.exists(sovits_path):
        alt_paths = [
            os.path.join(base_dir, "GPT_SoVITS", "SoVITS_weights", "白厄_extend_e12_s2664.pth"),
            os.path.join(base_dir, "SoVITS_weights", "白厄_extend_e12_s2664.pth")
        ]
        for path in alt_paths:
            if os.path.exists(path):
                sovits_path = path
                print(f"找到替代SoVITS模型路径: {sovits_path}")
                break
    
    if not os.path.exists(gpt_path):
        alt_paths = [
            os.path.join(base_dir, "GPT_SoVITS", "GPT_weights", "白厄_extend-e20.ckpt"),
            os.path.join(base_dir, "GPT_weights", "白厄_extend-e20.ckpt")
        ]
        for path in alt_paths:
            if os.path.exists(path):
                gpt_path = path
                print(f"找到替代GPT模型路径: {gpt_path}")
                break
    
    try:
        # 加载SoVITS模型
        sovits_result = load_sovits_model(sovits_path)
        print(f"sovits_result类型: {type(sovits_result)}")
        # 处理SoVITS模型加载结果
        if sovits_result is None:
            print("警告: load_sovits_model返回了None")
            _sovits_model = None
            _hps = None
        elif isinstance(sovits_result, tuple) and len(sovits_result) == 2:
            _sovits_model, _hps = sovits_result
        else:
            print(f"警告: load_sovits_model返回了意外格式: {sovits_result}")
            _sovits_model = sovits_result
            _hps = None
            
        # 加载GPT模型
        gpt_result = load_gpt_model(gpt_path)
        print(f"gpt_result类型: {type(gpt_result)}")
            
        # 处理GPT模型加载结果
        if gpt_result is None:
            print("警告: load_gpt_model返回了None")
            _gpt_model = None
            _dict_language = None
        elif isinstance(gpt_result, tuple) and len(gpt_result) == 2:
            _gpt_model, _dict_language = gpt_result
        else:
            print(f"警告: load_gpt_model返回了意外格式: {gpt_result}")
            _gpt_model = gpt_result
            _dict_language = None
            
        print("TTS模型加载完成")
        
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        import traceback
        traceback.print_exc()
        _sovits_model = None
        _gpt_model = None
        _hps = None
        _dict_language = None


def generate_voice(text, save_dir="voice"):
    """生成AI语音并保存到指定目录"""
    # 确保模型已加载
    init_models()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一文件名 - 确保在函数开头就定义 filepath
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"voice_{timestamp}.wav"
    filepath = os.path.join(save_dir, filename)
    
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 参考音频设置（使用绝对路径）
    ref_audio = os.path.join(base_dir, "Referance_voice", "【开心_happy】仪式开始前的这段时间尽可自由支配，好好休息吧…我也要花些时间整理思绪。回头见了，朋友们。.wav")
    ref_text = "仪式开始前的这段时间尽可自由支配，好好休息吧…我也要花些时间整理思绪。回头见了，朋友们。"
    
    # 参考音频列表（使用绝对路径）
    ref_audios = [
        os.path.join(base_dir, "Referance_voice", "【开心_happy】趁着天色正好，两位不如多走走看看吧。不打扰你们，回头见了。.wav"),
        os.path.join(base_dir, "Referance_voice", "【开心_happy】…不过，一直讨论这么苦大仇深的话题也属实扫兴。好像还是我起的头，真难为情。.wav"),
        os.path.join(base_dir, "Referance_voice", "【开心_happy】仪式开始前的这段时间尽可自由支配，好好休息吧…我也要花些时间整理思绪。回头见了，朋友们。.wav")
    ]
    
    # 检查引用文件是否存在
    print(f"主参考音频存在: {os.path.exists(ref_audio)}")
    for i, audio in enumerate(ref_audios):
        print(f"参考音频 {i+1} 存在: {os.path.exists(audio)}")
    
    try:
        # 调用GPT-SOVITS进行语音合成
        print(f"开始生成语音，使用参考音频: {ref_audio}")
        print(f"参考音频文件存在: {os.path.exists(ref_audio)}")
        
        wav_generator = get_tts_wav(
            ref_wav_path=ref_audio,
            prompt_text=ref_text,
            prompt_language="中文",
            text=text,
            text_language="中文",
            how_to_cut="凑四句一切",
            top_k=15,
            top_p=1,
            temperature=1,
            ref_free=False,
            speed=0.85,
            if_freeze=False,
            inp_refs=ref_audios
        )
        
        # 从生成器获取音频数据
        print("从生成器获取音频数据...")
        sample_rate = 24000
        audio_data = None
        
        # 获取生成器产生的所有数据
        for chunk in wav_generator:
            if isinstance(chunk, tuple) and len(chunk) >= 2:
                sample_rate, audio_data = chunk
        
        # 检查并保存音频数据
        if audio_data is not None:
            print(f"获取到音频数据，类型: {type(audio_data)}")
            
            # 如果是整数类型，转换为浮点数并归一化
            if np.issubdtype(audio_data.dtype, np.integer):
                audio_array = audio_data.astype(np.float32) / 32768.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
                
            # 使用已经定义的filepath保存
            sf.write(filepath, audio_array, sample_rate)
            print(f"音频已保存到: {filepath}")
            return filepath
        else:
            print("未获取到有效的音频数据")
            
    except Exception as e:
        print(f"处理音频时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 如果出错，生成空白音频文件作为备用
    print("生成备用空白音频文件")
    empty_audio = np.zeros(24000, dtype=np.float32)
    sf.write(filepath, empty_audio, 24000)
    return filepath  # 始终返回有效路径

def download_g2pw_model():
    """下载G2PW模型文件"""
    import urllib.request
    import zipfile
    
    g2pw_dir = os.environ.get('G2PW_MODEL_DIR', '')
    target_file = os.path.join(g2pw_dir, 'G2PWModel_1.1.zip')
    
    if os.path.exists(target_file):
        print("G2PW模型文件已存在，无需下载")
        return True
        
    try:
        print(f"开始下载G2PW模型到 {target_file}")
        url = "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip"
        urllib.request.urlretrieve(url, target_file)
        print("G2PW模型下载完成")
        return True
    except Exception as e:
        print(f"下载G2PW模型失败: {e}")
        return False