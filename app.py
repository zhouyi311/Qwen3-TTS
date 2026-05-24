import gradio as gr
import torch
import gc
import os
import re
import time
import shutil
import json
import soundfile as sf
import numpy as np
from huggingface_hub import try_to_load_from_cache, snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from qwen_tts import Qwen3TTSModel

# ==========================================
# 初始化 IO 目录系统
# ==========================================
IO_DIR = "io"
OUTPUT_DIR = os.path.join(IO_DIR, "outputs")
VOICE_LIB_DIR = os.path.join(IO_DIR, "voice_library")
MODELS_DIR = os.path.join(IO_DIR, "models")
SETTINGS_FILE = os.path.join(IO_DIR, "settings.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_LIB_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# 用户偏好设置持久化逻辑
# ==========================================
DEFAULT_SETTINGS = {
    "storage_option": "default",
    "cv_model_size": "1.7B",
    "vc_model_size": "1.7B"
}

def load_user_settings():
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    
    # 兼容处理：如果读到的是旧版的中文值，强制恢复为 default
    if settings["storage_option"] not in ["default", "local"]:
        settings["storage_option"] = "default"
    return settings

def save_user_setting(key, value):
    settings = load_user_settings()
    settings[key] = value
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to save settings {key}: {e}")

user_settings = load_user_settings()

# ==========================================
# 定义模型标识符映射
# ==========================================
MODEL_GROUPS = {
    "🪄 音色设计 (Voice Design)": {
        "Voice Design 1.7B (音色设计1.7B)": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    },
    "🎙️ 预设配音 (Custom Voice)": {
        "Custom Voice 1.7B (预设配音1.7B)": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Custom Voice 0.6B (预设配音0.6B)": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    },
    "👯 克隆音色 (Voice Clone)": {
        "Base 1.7B (克隆音色1.7B)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Base 0.6B (克隆音色0.6B)": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    }
}

ALL_MODELS = {k: v for group in MODEL_GROUPS.values() for k, v in group.items()}

MODEL_INFO = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "音色设计模型，通过自然语言描述（Prompt）创造全新音色。仅提供1.7B，无轻量版。",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "预设配音使用模型，内建多种预训练音色，支持内置发音人及情感指令控制。",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "同上，轻量级。速度快，显存占用低。",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "克隆音色与特征提取使用模型，3~10 秒参考音频即可复刻音色，15秒为最佳，过长则可能出现劣化。",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "同上，轻量级。注：0.6B与1.7B提取的特征（音色库）互不兼容。"
}

class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model = None

    def load_model(self, target_model_id, storage_option):
        """
        根据指定的 HF repo_id 加载对应的模型权重。
        """
        # 动态设定环境变量强制修改底层所有依赖(HF/ModelScope)的下载及读取行为
        if storage_option == "local":
            os.environ["HF_HUB_CACHE"] = os.path.abspath(MODELS_DIR)
            os.environ["MODELSCOPE_CACHE"] = os.path.abspath(MODELS_DIR)
        else:
            os.environ.pop("HF_HUB_CACHE", None)
            os.environ.pop("MODELSCOPE_CACHE", None)

        if not target_model_id:
            raise ValueError("未指定需要加载的模型。")
            
        if check_status(target_model_id, storage_option) == "❌ 未下载":
            raise ValueError(f"模型【{target_model_id}】尚未下载！请先前往“⚙️ 模型管理”页面进行下载。")

        if self.current_model_name == target_model_id and getattr(self, "current_storage_option", None) == storage_option:
            return f"✅ 模型 [{target_model_id}] 已就绪。"

        # 释放旧模型显存
        if self.model is not None:
            print(f"\n[Model Switch] Unloading previous model and releasing VRAM: {self.current_model_name} ...")
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.model = None

        try:
            print(f"\n[Model Load] Loading model: {target_model_id} (this may take a while) ...")
            # 若硬件不支持 flash_attention_2，可将其注释或设为 eager/sdpa
            cache_dir = os.path.abspath(MODELS_DIR) if storage_option == "local" else None
            self.model = Qwen3TTSModel.from_pretrained(
                target_model_id,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                cache_dir=cache_dir,
                local_files_only=True # 绝对禁止在生成时进行静默网络下载
            )
            self.current_model_name = target_model_id
            self.current_storage_option = storage_option
            print(f"[Model Load] Model {target_model_id} loaded successfully!")
            return f"✅ 已加载模型: {target_model_id}"
        except Exception as e:
            self.current_model_name = None
            raise RuntimeError(f"模型加载失败: {str(e)}")

manager = ModelManager()

# ==========================================
# 辅助工具函数
# ==========================================
def audio_to_int16(wav):
    """将 float32 音频转换为 int16 格式，消除 Gradio 播放器的 UserWarning"""
    if wav is None or len(wav) == 0:
        return np.zeros(1, dtype=np.int16)
    return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

def safe_name(text, max_len=None):
    """移除文件名中的非法字符及空白符，并限制最大长度"""
    if not text:
        return "empty"
    clean_text = re.sub(r'[\\/*?:"<>|\s]', "", str(text))
    if max_len:
        clean_text = clean_text[:max_len]
    return clean_text

def save_audio_to_io(prefix, sr, wav):
    """统一将生成的音频持久化保存到 io/outputs/ 目录下"""
    filename = f"{prefix}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    sf.write(filepath, wav, sr)
    return filepath

def get_model_size_from_prompt(pt_path):
    """探测 .pt 文件是由 1.7B 还是 0.6B 提取的（通过 hidden_size 1024 vs 2048 判断）"""
    try:
        data = torch.load(pt_path, weights_only=False, map_location="cpu")
        def search_shape(obj):
            if isinstance(obj, torch.Tensor):
                if len(obj.shape) >= 1:
                    if obj.shape[-1] == 2048: return "1.7B"
                    if obj.shape[-1] == 1024: return "0.6B"
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    res = search_shape(item)
                    if res: return res
            elif isinstance(obj, dict):
                for item in obj.values():
                    res = search_shape(item)
                    if res: return res
            return None
        return search_shape(data)
    except Exception:
        return None

def get_saved_voices():
    """获取 io/voice_library/ 下所有已保存的音色列表"""
    return [f[:-3] for f in os.listdir(VOICE_LIB_DIR) if f.endswith(".pt")]

def check_status(repo_id, storage_option):
    cache_dir = os.path.abspath(MODELS_DIR) if storage_option == "local" else None
    filepath = try_to_load_from_cache(repo_id, "config.json", cache_dir=cache_dir)
    if isinstance(filepath, str):
        return "✅ 已下载"
    return "❌ 未下载"

def action_download(repo_id, storage_option):
    cache_dir = os.path.abspath(MODELS_DIR) if storage_option == "local" else None
    try:
        snapshot_download(repo_id, cache_dir=cache_dir)
        return "✅ 已下载"
    except Exception as e:
        return f"❌ 下载失败 ({str(e)})"

def action_delete(repo_id, storage_option):
    cache_dir = os.path.abspath(MODELS_DIR) if storage_option == "local" else HUGGINGFACE_HUB_CACHE
    folder_name = "models--" + repo_id.replace("/", "--")
    target_path = os.path.join(cache_dir, folder_name)
    if os.path.exists(target_path):
        try:
            shutil.rmtree(target_path)
            return "❌ 未下载"
        except Exception as e:
            return f"⚠️ 删除失败 ({str(e)})"
    return "❌ 未下载"

def build_status_updates(cv_size, vc_size, lib_size):
    """构造四个 Tab 下各自独立的模型匹配状态消息"""
    
    loaded = manager.current_model_name
    if not loaded:
        msg = "未加载... (点击生成按钮时将自动加载对应模型)"
        return msg, msg, msg, msg
    
    def fmt(target):
        loaded_short = loaded.split('/')[-1]
        target_short = target.split('/')[-1]
        if loaded == target:
            return f"🟢 已就绪 (当前匹配: {loaded_short})，可极速生成。"
        else:
            return f"🟡 待切换 (当前加载: {loaded_short} | 本任务需: {target_short})，生成时将自动重载并额外耗时。"
    
    cv_target = f"Qwen/Qwen3-TTS-12Hz-{cv_size}-CustomVoice"
    vd_target = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    vc_target = f"Qwen/Qwen3-TTS-12Hz-{vc_size}-Base"
    lib_target = f"Qwen/Qwen3-TTS-12Hz-{lib_size}-Base"
    
    return (
        fmt(cv_target),
        fmt(vd_target),
        fmt(vc_target),
        fmt(lib_target) 
    )

# ==========================================
# 任务与模型分离逻辑 (先加载更新状态，再生成音频)
# ==========================================
def pre_load_cv(text, model_size, storage_option, cv_size, vc_size, lib_size):
    if not text.strip(): raise gr.Error("请输入合成文本。")
    manager.load_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-CustomVoice", storage_option)
    return build_status_updates(cv_size, vc_size, lib_size)

def pre_load_vd(text, instruct, storage_option, cv_size, vc_size, lib_size):
    if not text.strip() or not instruct.strip(): raise gr.Error("合成文本和声音描述都不能为空。")
    manager.load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", storage_option)
    return build_status_updates(cv_size, vc_size, lib_size)

def pre_load_vc(text, ref_audio, ref_text, model_size, storage_option, cv_size, vc_size, lib_size):
    if not text.strip(): raise gr.Error("请输入目标合成文本。")
    if not ref_audio: raise gr.Error("请上传参考音频。")
    if not ref_text.strip(): raise gr.Error("请输入参考音频对应的文本。")
    manager.load_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base", storage_option)
    return build_status_updates(cv_size, vc_size, lib_size)

def pre_load_save_feature(voice_name, model_size, storage_option, cv_size, vc_size, lib_size):
    if not voice_name.strip(): raise gr.Error("请输入音色名称。")
    manager.load_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base", storage_option)
    return build_status_updates(cv_size, vc_size, lib_size)

def pre_load_lib(voice_name, text, model_size, storage_option, cv_size, vc_size, lib_size):
    if not voice_name: raise gr.Error("请选择一个音色。")
    lines = [line.strip() for line in text.split('///') if line.strip()]
    if not lines: raise gr.Error("请输入有效的合成文本。")
    
    # 自动探测特征文件的模型维度，防止 1024 vs 2048 的 tensor 合并报错
    pt_path = os.path.join(VOICE_LIB_DIR, f"{voice_name}.pt")
    if os.path.exists(pt_path):
        detected_size = get_model_size_from_prompt(pt_path)
        if detected_size and detected_size != model_size:
            model_size = detected_size
            gr.Info(f"探测到该音色提取自 {detected_size} 模型，后台已临时切换匹配以完成本次生成！")
            
    manager.load_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base", storage_option)
    st_cv, st_vd, st_vc, st_lib = build_status_updates(cv_size, vc_size, model_size)
    return st_cv, st_vd, st_vc, st_lib, gr.update(value=model_size)

# ==========================================
# 功能执行函数 (纯生成，不再包含模型加载与全局 UI 状态更新)
# ==========================================
def generate_custom_voice(text, language, speaker, instruct):
    print(f"\n[Task Start] Custom Voice (Speaker: {speaker})")
    wavs, sr = manager.model.generate_custom_voice(text=text, language=language, speaker=speaker, instruct=instruct)
    time_str = time.strftime("%y%m%d_%H%M%S")
    prefix = f"CustomVoice_{time_str}_{safe_name(speaker)}_{safe_name(text, 2)}"
    filepath = save_audio_to_io(prefix, sr, wavs[0])
    print(f"[Task Complete] Successfully generated and saved to: {filepath}")
    return (sr, audio_to_int16(wavs[0])), filepath, text, gr.update(interactive=True)

def generate_voice_design(text, language, instruct):
    print(f"\n[Task Start] Voice Design")
    wavs, sr = manager.model.generate_voice_design(text=text, language=language, instruct=instruct)
    time_str = time.strftime("%y%m%d_%H%M%S")
    prefix = f"VoiceDesign_{time_str}_{safe_name(instruct, 10)}_{safe_name(text, 2)}"
    filepath = save_audio_to_io(prefix, sr, wavs[0])
    print(f"[Task Complete] Successfully generated and saved to: {filepath}")
    return (sr, audio_to_int16(wavs[0])), filepath, text, gr.update(interactive=True)

def generate_voice_clone(text, language, ref_audio, ref_text):
    print(f"\n[Task Start] Voice Clone")
    wavs, sr = manager.model.generate_voice_clone(text=text, language=language, ref_audio=ref_audio, ref_text=ref_text)
    time_str = time.strftime("%y%m%d_%H%M%S")
    prefix = f"VoiceClone_{time_str}_{safe_name(ref_text, 10)}_{safe_name(text, 2)}"
    filepath = save_audio_to_io(prefix, sr, wavs[0])
    print(f"[Task Complete] Successfully generated and saved to: {filepath}")
    return (sr, audio_to_int16(wavs[0])), gr.update(interactive=True)

def save_voice_feature(voice_name, ref_audio, ref_text):
    voice_name = voice_name.strip()
    model_name = manager.current_model_name or ""
    suffix = "_06" if "0.6B" in model_name else "_17"
    if voice_name.endswith("_17") or voice_name.endswith("_06"):
        voice_name = voice_name[:-3]
    voice_name += suffix
        
    print(f"\n[Task Start] Extracting and saving voice prompt: {voice_name}")
    prompt_items = manager.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    save_path = os.path.join(VOICE_LIB_DIR, f"{voice_name}.pt")
    torch.save(prompt_items, save_path)
    print(f"[Task Complete] Voice [{voice_name}] saved successfully.")
    
    display_model = "0.6B" if "0.6B" in model_name else "1.7B"
    msg = f"✅ 音色 [{voice_name}] 保存成功！\n注：已使用 {display_model}-Base 固化特征。"
    return gr.update(value=msg, visible=True), gr.update(choices=get_saved_voices())

def generate_from_library(voice_name, text, language, batch_size):
    """读取提取好的 prompt 直接批量生成声音"""
    prompt_items = torch.load(os.path.join(VOICE_LIB_DIR, f"{voice_name}.pt"), weights_only=False)
    
    batch_size = int(batch_size)
    lines = []
    lines = [line.strip() for line in text.split('///') if line.strip()]
    if not lines:
        raise gr.Error("请输入有效的合成文本。")
    if len(lines) > 10:
        gr.Warning("切分段数最多支持 10 段，超出部分已被忽略。")
        lines = lines[:10]

    # 核心修复：使用列表推导式避免浅拷贝造成的 UI 状态互相覆盖
    results = [gr.update(value=None, visible=False) for _ in range(10)]
    
    start_time = time.time()
    total_batches = (len(lines) + batch_size - 1) // batch_size
    
    print(f"\n[Task Start] Batch Generation (Voice: {voice_name} | Batch size: {batch_size})")
    print(f"[Progress] Total {len(lines)} segments, will be executed in {total_batches} batches.")
    yield tuple(results + [f"⏳ 开始生成，共 {total_batches} 个任务..."])

    global_idx = 0
    for batch_idx, i in enumerate(range(0, len(lines), batch_size)):
        print(f"[Progress] Executing batch {batch_idx + 1}/{total_batches} ...")
        chunk_lines = lines[i:i + batch_size]
        chunk_wavs, chunk_sr = manager.model.generate_voice_clone(text=chunk_lines, language=language, voice_clone_prompt=prompt_items)
        
        for wav in chunk_wavs:
            # 核心修复：防止模型因文本过短等原因生成空音频，导致前端 Audio 播放器卡死无限 Loading
            if len(wav) == 0:
                wav = np.zeros(int(chunk_sr * 0.5), dtype=np.float32)
                
            line = lines[global_idx]
            time_str = time.strftime("%y%m%d_%H%M%S")
            prefix = f"Asset_{time_str}_s{global_idx+1:02d}_{safe_name(voice_name)}_{safe_name(line, 2)}"
            save_audio_to_io(prefix, chunk_sr, wav)
            results[global_idx] = gr.update(value=(chunk_sr, audio_to_int16(wav)), visible=True, label=f"Section {global_idx+1}")
            global_idx += 1
            
        remaining = total_batches - batch_idx - 1
        print(f"[Progress] Batch {batch_idx + 1}/{total_batches} completed.")
        if remaining > 0:
            batch_status = f"⏳ 正在生成剩余音频（剩余 {remaining} 个任务）..."
        else:
            elapsed = time.time() - start_time
            batch_status = f"✅ 已全部完成，总耗时 {elapsed:.1f} 秒。"
            print(f"[Task Complete] Batch generation finished in {elapsed:.1f} seconds.\n")
            
        # 每生成完一个批次，立即更新到前端，实现渐进式流式展现！
        yield tuple(results + [batch_status])
        
        # 顺手清理一下当前批次的显存碎片，为下一批次腾出空间
        torch.cuda.empty_cache()

# ==========================================
# Gradio UI 界面构建
# ==========================================
with gr.Blocks(title="Qwen3-TTS WebUI") as app:
    gr.Markdown("# 🚀 Qwen3-TTS 综合语音控制台")
    
    # 功能 Tabs 区域
    with gr.Tabs():
        # Tab 1: 模型管理
        with gr.TabItem("⚙️ 模型管理"):
            
            with gr.Row():
                with gr.Column(variant="panel", scale=5):
                    gr.Markdown("🗂️ 模型下载与存储设置 *（预设存储为系统缓存。可隔离存储到项目本地 `io/models`）*")
                    storage_option = gr.Radio(
                        choices=[
                            ("默认系统缓存(user目录下)", "default"), 
                            ("项目本地 (io/models文件夹)", "local")
                        ], 
                        value=user_settings["storage_option"], 
                        label="当前选用路径",
                        interactive=True
                    )
                with gr.Column(variant="panel", scale=4):
                    gr.Markdown("🎛️ 默认模型版本选择 *(1.7B 质量更高，0.6B 速度快)*")
                    with gr.Row():
                        cv_model_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["cv_model_size"], label="预设配音模型")
                        vc_model_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["vc_model_size"], label="克隆音色模型")
            
            status_boxes = []
            
            for group_name, models in MODEL_GROUPS.items():
                with gr.Row():
                    with gr.Column(variant="panel"):
                        gr.Markdown(f"#### {group_name}模型")
                        for display_name, repo_id in models.items():
                            with gr.Row(variant="panel"):
                                with gr.Column(scale=5, min_width=280):
                                    desc = MODEL_INFO.get(repo_id, "")
                                    gr.Markdown(f"**{display_name}**<br>`{repo_id}`<br>*{desc}*")
                                
                                status_box = gr.Textbox(label="下载状态", value="检查中...", interactive=False, lines=1, max_lines=1, scale=2, min_width=120)
                                status_boxes.append(status_box)
                                
                                with gr.Column(scale=1, min_width=100):
                                    dl_btn = gr.Button("⬇️ 下载", variant="primary")
                                    del_btn = gr.Button("🗑️ 删除", variant="stop")
                                
                                # 点击事件绑定
                                dl_btn.click(fn=lambda s, r=repo_id: action_download(r, s), inputs=[storage_option], outputs=[status_box])
                                del_btn.click(fn=lambda s, r=repo_id: action_delete(r, s), inputs=[storage_option], outputs=[status_box])
            
            # 当切换存储位置时，刷新全部状态
            storage_option.change(
                fn=lambda s: [check_status(repo_id, s) for _, repo_id in ALL_MODELS.items()],
                inputs=[storage_option],
                outputs=status_boxes
            )

        # Tab 2: 预设配音
        with gr.TabItem("🎙️ 预设配音"):
            with gr.Row():
                cv_load_status = gr.Textbox(label="🖥️ 模型状态", value="未加载... (点击生成按钮时将自动加载对应模型)", interactive=False, lines=1, max_lines=1, scale=4)
                cv_current_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["cv_model_size"], label="当前配音模型", scale=1)
            with gr.Row():
                with gr.Column():
                    cv_text = gr.Textbox(label="合成文本", placeholder="输入你想让模型说的话...")
                    cv_lang = gr.Dropdown(choices=["Auto", "Chinese", "English", "Japanese", "Korean"], value="Auto", label="语言 (推荐 Auto)")
                    cv_speaker = gr.Dropdown(choices=["Vivian", "Ryan", "Uncle_Fu", "Dylan", "Eric", "Aiden", "Ono_Anna", "Sohee"], value="Vivian", label="发音人")
                    cv_instruct = gr.Textbox(label="情感/语气指令 (选填)", placeholder="例如：用非常生气的语气说")
                    cv_btn = gr.Button("生成语音", variant="primary")
                with gr.Column():
                    cv_output = gr.Audio(label="输出音频", interactive=False)
                    # 常驻的保存区域，默认按钮不可点击 (生成后激活)
                    with gr.Group():
                        gr.Markdown("#### 💾 克隆当前音频特征并存至音色库")
                        with gr.Row():
                            cv_save_name = gr.Textbox(label="保存音色名称", placeholder="例如：愤怒的Vivian_01", lines=1, max_lines=1, scale=5)
                            cv_save_model_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["vc_model_size"], label="克隆模型", scale=0, min_width=220)
                        with gr.Row():
                            cv_save_btn = gr.Button("克隆并存至音色库", variant="secondary", interactive=False, scale=2)
                        cv_save_status = gr.Textbox(show_label=False, interactive=False, lines=2, visible=False)
                        cv_audio_path_state = gr.State()
                        cv_text_state = gr.State()

        # Tab 3: 音色设计
        with gr.TabItem("🪄 音色设计"):
            vd_load_status = gr.Textbox(label="🖥️ 模型状态", value="未加载... (点击生成按钮时将自动加载对应模型)", interactive=False, lines=1, max_lines=1)
            with gr.Row():
                with gr.Column():
                    vd_text = gr.Textbox(label="合成文本", placeholder="输入你想让模型说的话...")
                    vd_lang = gr.Dropdown(choices=["Auto", "Chinese", "English", "Japanese", "Korean"], value="Auto", label="语言")
                    vd_instruct = gr.Textbox(label="声音描述 (Prompt)", placeholder="例如：体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显...", lines=3)
                    vd_btn = gr.Button("捏造声音并生成", variant="primary")
                with gr.Column():
                    vd_output = gr.Audio(label="输出音频", interactive=False)
                    # 常驻的保存区域，默认按钮不可点击 (生成后激活)
                    with gr.Group():
                        gr.Markdown("#### 💾 克隆当前音频特征并存至音色库")
                        with gr.Row():
                            vd_save_name = gr.Textbox(label="保存音色名称", placeholder="例如：傲娇萝莉_01", lines=1, max_lines=1, scale=5)
                            vd_save_model_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["vc_model_size"], label="克隆模型", scale=0, min_width=220)
                        with gr.Row():
                            vd_save_btn = gr.Button("克隆并存至音色库", variant="secondary", interactive=False, scale=2)
                        vd_save_status = gr.Textbox(show_label=False, interactive=False, lines=2, visible=False)
                        vd_audio_path_state = gr.State()
                        vd_text_state = gr.State()

        # Tab 4: 克隆音色
        with gr.TabItem("👯 克隆音色"):
            with gr.Row():
                vc_load_status = gr.Textbox(label="🖥️ 模型状态", value="未加载... (点击生成按钮时将自动加载对应模型)", interactive=False, lines=1, max_lines=1, scale=4)
                vc_current_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["vc_model_size"], label="当前克隆模型", scale=1)
            with gr.Row():
                with gr.Column():
                    vc_ref_audio = gr.Audio(label="参考音频 (推荐 3-10 秒的清晰人声)", type="filepath")
                    vc_ref_text = gr.Textbox(label="参考音频对应的文本", placeholder="请准确输入上方参考音频中所说的内容...")
                    vc_text = gr.Textbox(label="目标合成文本", placeholder="输入你想让模型说的话...", lines=3)
                    vc_lang = gr.Dropdown(choices=["Auto", "Chinese", "English", "Japanese", "Korean"], value="Auto", label="语言")
                    vc_btn = gr.Button("克隆并生成", variant="primary")
                with gr.Column():
                    vc_output = gr.Audio(label="输出音频", interactive=False)
                    # 常驻的保存区域，默认按钮不可点击 (生成后激活)
                    with gr.Group():
                        gr.Markdown("#### 💾 克隆当前音频特征并存至音色库")
                        with gr.Row():
                            vc_save_name = gr.Textbox(label="保存音色名称", placeholder="例如：深情大叔_01", lines=1, max_lines=1, scale=5)
                            vc_save_model_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["vc_model_size"], label="克隆模型", scale=0, min_width=220)
                        with gr.Row():
                            vc_save_btn = gr.Button("克隆并存至音色库", variant="secondary", interactive=False, scale=2)
                        vc_save_status = gr.Textbox(show_label=False, interactive=False, lines=2, visible=False)

        # Tab 5: 声音资产库
        with gr.TabItem("🗂️ 我的音色库"):
            with gr.Row():
                lib_load_status = gr.Textbox(label="🖥️ 模型状态", value="未加载... (点击生成按钮时将自动加载对应模型)", interactive=False, lines=1, max_lines=1, scale=4)
                lib_current_size = gr.Radio(choices=["1.7B", "0.6B"], value=user_settings["vc_model_size"], label="当前生成模型", scale=1)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        lib_voice_dropdown = gr.Dropdown(choices=get_saved_voices(), label="选择我的声音", scale=4)
                        lib_refresh_btn = gr.Button("🔄 刷新", scale=1, min_width=100)
                    lib_text = gr.Textbox(label="目标合成文本", placeholder="输入你想让模型说的话...\n若文本中包含 ///，系统将以此为界自动切分为多段音频。", lines=4)
                    with gr.Row():
                        lib_lang = gr.Dropdown(choices=["Auto", "Chinese", "English", "Japanese", "Korean"], value="Auto", label="语言")
                        lib_batch_size = gr.Dropdown(choices=["1", "2", "3", "4"], value="1", label="多线程并发")
                    lib_gen_btn = gr.Button("使用该音色生成", variant="primary")
                with gr.Column():
                    lib_batch_outputs = []
                    for i in range(10):
                        lib_batch_outputs.append(
                            gr.Audio(label=f"输出音频 - 分段 {i+1}", interactive=False, visible=(i==0))
                        )
                    lib_batch_status = gr.Textbox(label="批量生成进度", value="就绪", interactive=False, lines=1, max_lines=1)
                    
    # 底部全局事件绑定
    cv_btn.click(
        pre_load_cv, inputs=[cv_text, cv_current_size, storage_option, cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status]
    ).success(
        generate_custom_voice, inputs=[cv_text, cv_lang, cv_speaker, cv_instruct], outputs=[cv_output, cv_audio_path_state, cv_text_state, cv_save_btn]
    )
    vd_btn.click(
        pre_load_vd, inputs=[vd_text, vd_instruct, storage_option, cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status]
    ).success(
        generate_voice_design, inputs=[vd_text, vd_lang, vd_instruct], outputs=[vd_output, vd_audio_path_state, vd_text_state, vd_save_btn]
    )
    vc_btn.click(
        pre_load_vc, inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_current_size, storage_option, cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status]
    ).success(
        generate_voice_clone, inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text], outputs=[vc_output, vc_save_btn]
    )

    cv_save_btn.click(
        pre_load_save_feature, inputs=[cv_save_name, cv_save_model_size, storage_option, cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status]
    ).success(
        save_voice_feature, inputs=[cv_save_name, cv_audio_path_state, cv_text_state], outputs=[cv_save_status, lib_voice_dropdown]
    )
    vd_save_btn.click(
        pre_load_save_feature, inputs=[vd_save_name, vd_save_model_size, storage_option, cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status]
    ).success(
        save_voice_feature, inputs=[vd_save_name, vd_audio_path_state, vd_text_state], outputs=[vd_save_status, lib_voice_dropdown]
    )
    vc_save_btn.click(
        pre_load_save_feature, inputs=[vc_save_name, vc_save_model_size, storage_option, cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status]
    ).success(
        save_voice_feature, inputs=[vc_save_name, vc_ref_audio, vc_ref_text], outputs=[vc_save_status, lib_voice_dropdown]
    )
    lib_refresh_btn.click(lambda: gr.update(choices=get_saved_voices()), inputs=[], outputs=[lib_voice_dropdown])
    lib_gen_btn.click(
        pre_load_lib, inputs=[lib_voice_dropdown, lib_text, lib_current_size, storage_option, cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status, lib_current_size]
    ).success(
        generate_from_library, inputs=[lib_voice_dropdown, lib_text, lib_lang, lib_batch_size], outputs=lib_batch_outputs + [lib_batch_status]
    )

    def sync_status(cv, vc, lib):
        return build_status_updates(cv, vc, lib)

    cv_current_size.change(fn=sync_status, inputs=[cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status])
    vc_current_size.change(fn=sync_status, inputs=[cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status])
    lib_current_size.change(fn=sync_status, inputs=[cv_current_size, vc_current_size, lib_current_size], outputs=[cv_load_status, vd_load_status, vc_load_status, lib_load_status])

    def change_global_cv(v):
        save_user_setting("cv_model_size", v)
        return v
    
    def change_global_vc(v):
        save_user_setting("vc_model_size", v)
        return v, v, v, v, v

    # 保存用户偏好设置的隐式事件绑定
    storage_option.change(fn=lambda v: save_user_setting("storage_option", v), inputs=[storage_option])
    cv_model_size.change(fn=change_global_cv, inputs=[cv_model_size], outputs=[cv_current_size])
    vc_model_size.change(fn=change_global_vc, inputs=[vc_model_size], outputs=[vc_current_size, lib_current_size, cv_save_model_size, vd_save_model_size, vc_save_model_size])

    # 启动时自动触发一次状态检查
    app.load(
        fn=lambda s: [check_status(repo_id, s) for _, repo_id in ALL_MODELS.items()],
        inputs=[storage_option],
        outputs=status_boxes
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=8004, inbrowser=True, theme=gr.themes.Soft())
