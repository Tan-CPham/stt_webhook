#!/usr/bin/env python3
"""
Stereo to 2 Mono - Tách file stereo WAV thành 2 file mono
File stereo sẽ được tách thành:
- agent.wav (kênh trái)
- customer.wav (kênh phải)
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

def stereo_to_mono(input_file, output_dir=None):
    """
    Tách file stereo thành 2 file mono
    
    Args:
        input_file (str): Đường dẫn file stereo input
        output_dir (str): Thư mục output (mặc định cùng thư mục với input)
    
    Returns:
        tuple: (agent_file_path, customer_file_path)
    """
    
    # Kiểm tra file input
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Không tìm thấy file: {input_file}")
    
    print(f"📁 Đang đọc file stereo: {input_file}")
    
    # Đọc file audio
    try:
        audio_data, sample_rate = sf.read(input_file)
        print(f"✅ Sample rate: {sample_rate}Hz")
        print(f"📊 Shape: {audio_data.shape}")
        
        # Kiểm tra xem có phải stereo không
        if len(audio_data.shape) == 1:
            print("⚠️  File này là mono, không thể tách thành 2 kênh")
            return None, None
        
        if audio_data.shape[1] != 2:
            print(f"⚠️  File này có {audio_data.shape[1]} kênh, chỉ hỗ trợ stereo (2 kênh)")
            return None, None
            
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None, None
    
    # Xác định thư mục output
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tách 2 kênh
    left_channel = audio_data[:, 0]   # Kênh trái -> Agent
    right_channel = audio_data[:, 1]  # Kênh phải -> Customer
    
    # Tạo tên file output
    input_name = Path(input_file).stem
    agent_file = os.path.join(output_dir, f"{input_name}_agent.wav")
    customer_file = os.path.join(output_dir, f"{input_name}_customer.wav")
    
    print(f"🔄 Đang tách stereo thành 2 file mono...")
    
    # Lưu file mono
    try:
        # Lưu kênh agent (trái)
        sf.write(agent_file, left_channel, sample_rate)
        print(f"✅ Đã lưu agent: {agent_file}")
        
        # Lưu kênh customer (phải)
        sf.write(customer_file, right_channel, sample_rate)
        print(f"✅ Đã lưu customer: {customer_file}")
        
        # Thống kê
        print(f"\n📊 Thống kê:")
        print(f"   Original: {len(audio_data)/sample_rate:.2f}s, {audio_data.shape[1]} kênh")
        print(f"   Agent: {len(left_channel)/sample_rate:.2f}s, mono")
        print(f"   Customer: {len(right_channel)/sample_rate:.2f}s, mono")
        
        return agent_file, customer_file
        
    except Exception as e:
        print(f"❌ Lỗi lưu file: {e}")
        return None, None

def main():
    print("🎧 Stereo to 2 Mono Converter")
    print("=" * 50)
    
    # Kiểm tra đối số đầu vào
    if len(sys.argv) < 2:
        print("Cách sử dụng: python sterio_to_2mono.py <đường_dẫn_file_stereo> [thư_mục_output]")
        print("\nVí dụ:")
        print("  python sterio_to_2mono.py audio/call_recording.wav")
        print("  python sterio_to_2mono.py audio/call_recording.wav output/")
        return
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Thực hiện tách file
    try:
        agent_file, customer_file = stereo_to_mono(input_file, output_dir)
        
        if agent_file and customer_file:
            print("\n🎉 " + "="*40)
            print("HOÀN THÀNH!")
            print("="*44)
            print(f"🎤 Agent file: {agent_file}")
            print(f"👤 Customer file: {customer_file}")
            print("="*44)
        else:
            print("\n❌ Tách file thất bại!")
            
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

def batch_convert(input_dir, output_dir=None):
    """
    Tách tất cả file stereo WAV trong thư mục
    
    Args:
        input_dir (str): Thư mục chứa file stereo
        output_dir (str): Thư mục output
    """
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "mono_output")
    
    # Tìm tất cả file WAV
    wav_files = list(Path(input_dir).glob("*.wav"))
    
    if not wav_files:
        print(f"❌ Không tìm thấy file WAV nào trong {input_dir}")
        return
    
    print(f"🔍 Tìm thấy {len(wav_files)} file WAV")
    
    success_count = 0
    for wav_file in wav_files:
        print(f"\n--- Xử lý {wav_file.name} ---")
        try:
            agent_file, customer_file = stereo_to_mono(str(wav_file), output_dir)
            if agent_file and customer_file:
                success_count += 1
        except Exception as e:
            print(f"❌ Lỗi xử lý {wav_file.name}: {e}")
    
    print(f"\n🎉 Hoàn thành! Đã tách thành công {success_count}/{len(wav_files)} file")

if __name__ == "__main__":
    # Nếu muốn batch convert, uncomment dòng dưới
    # batch_convert("audio/", "output/")
    
    main()
