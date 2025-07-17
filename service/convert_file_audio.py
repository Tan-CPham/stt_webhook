from pydub import AudioSegment
import os

def convert_cgi_to_wav(cgi_filepath, output_wav_filepath):
    """
    Chuyển đổi một file audio .cgi sang định dạng .wav.

    Args:
        cgi_filepath (str): Đường dẫn đến file .cgi đầu vào.
        output_wav_filepath (str): Đường dẫn để lưu file .wav đầu ra.
    """
    try:
        # Pydub sẽ tự động xác định định dạng audio từ nội dung file
        # Ngay cả khi file có đuôi là .cgi
        print(f"Đang đọc file: {cgi_filepath}...")
        audio = AudioSegment.from_file(cgi_filepath)

        # Xuất file audio sang định dạng WAV
        print(f"Đang xuất sang định dạng WAV tại: {output_wav_filepath}...")
        audio.export(output_wav_filepath, format="wav")

        print("Chuyển đổi thành công!")
        return True

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        print("Vui lòng kiểm tra xem file .cgi có thực sự chứa dữ liệu audio hợp lệ không,")
        print("và đảm bảo ffmpeg đã được cài đặt và có trong PATH của hệ thống.")
        return False

# --- Ví dụ sử dụng ---

# 1. Chuyển đổi một file duy nhất
input_file = "path/to/your/audio.cgi"
output_file = "path/to/your/output.wav"

output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

convert_cgi_to_wav(input_file, output_file)


input_folder = "E:/zomzem_web/backend/speech_to_text/audio"
output_folder = "E:/zomzem_web/backend/speech_to_text/audio"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".cgi"):
        cgi_path = os.path.join(input_folder, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_folder, wav_filename)
        
        print("-" * 20)
        convert_cgi_to_wav(cgi_path, wav_path)