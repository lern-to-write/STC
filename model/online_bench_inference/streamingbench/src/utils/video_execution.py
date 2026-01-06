# 导入操作系统相关功能的模块
import os
# 导入ffmpeg-python库，用于视频处理
import ffmpeg

def split_video(video_file, start_time, end_time):
    """
    根据时间戳将视频分割成前缀部分。
    video_file: 视频文件的路径
    start_time: 开始时间（以秒为单位）
    end_time: 结束时间（以秒为单位）
    """
    # 提取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    # 构建输出目录路径，在视频文件同级目录下创建tmp_60文件夹
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    # 如果输出目录不存在，则创建该目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 构建输出文件路径，文件名包含原文件名和时间戳信息
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    # 检查输出文件是否已存在，如果存在则直接返回该文件路径
    if os.path.exists(output_file):
        print(f"Video file {output_file} already exists.")
        return output_file

    # 使用ffmpeg尝试分割视频
    try:
        (
            # 设置输入文件和开始时间
            ffmpeg
            .input(video_file, ss=int(start_time))
            # 设置输出文件、持续时间以及视频和音频编码格式
            .output(output_file, t=(int(end_time) - int(start_time)), vcodec='libx264', acodec='aac')
            # 覆盖已存在的输出文件
            .overwrite_output()
            # 执行命令，捕获标准输出和错误输出
            .run(capture_stdout=True, capture_stderr=True)
        )
    # 捕获ffmpeg执行过程中的错误
    except ffmpeg.Error as e:
        # 打印ffmpeg错误信息
        print(f"ffmpeg error: {e.stderr.decode('utf-8')}")
        
    # 打印视频分割完成的信息并返回输出文件路径
    print(f"Video: {output_file} splitting completed.")
    return output_file