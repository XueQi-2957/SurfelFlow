#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path

# 锚定工作目录为本脚本所在的根目录
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

def create_and_organize_dataset(input_dir, target_base_dir, scene_name):
    """
    检查输入目录并提取所有图片，强制将其重新归类组织为 GFlow 的多层级硬约束，
    也就是 target_base_dir/scene_name/scene_name/scene_name。
    """
    input_path = Path(input_dir).resolve()
    target_path = Path(target_base_dir).resolve()
    
    # 构建兼容的嵌套保存深层目录结构，例如：./dataset/MyScene/MyScene/MyScene/
    final_leaf_dir = target_path / scene_name / scene_name / scene_name
    
    # 这是传给后续各大脚本的核心参数父目录层级，例如：./dataset/MyScene
    parent_cmd_dir = target_path / scene_name
    
    print(f"[准备阶段] 正在解析你杂乱的源图片目录: {input_path}")
    print(f"[准备阶段] 即将归一化并挂载到最终资源树: {final_leaf_dir}")
    
    # 提取输入目录下所有的常见图像特征（并无视原有的单层复杂杂乱目录）
    img_exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
    img_files = []
    # 采取 rglob 支持你胡乱多层嵌套扔进来的大杂烩，都会被拉平成一层
    for ext in img_exts:
        img_files.extend(list(input_path.rglob(ext)))
        
    if not img_files:
        print(f"❌ 严重错误: 在目标源输入路径 '{input_path}' 及其内部完全找不到任何可用图片！")
        sys.exit(1)
        
    # 创造标准三级兼容目录
    final_leaf_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[合并阶段] 共扫出 {len(img_files)} 张图像，正在实施超光速符号链接/硬盘拷贝灌注...")
    for img_file in img_files:
        dest_file = final_leaf_dir / img_file.name
        # 为了不撑爆硬盘容量，第一顺位选用 Unix 系统内核级软链接去重建虚拟目录
        try:
            if not dest_file.exists():
                os.symlink(img_file, dest_file)
        except OSError:
            # 如果你把工程弄到了各种奇葩的跨卷 NTFS 上，则降级为老老实实的物理文件拷贝
            if not dest_file.exists():
                shutil.copy2(img_file, dest_file)
                
    print("[准备完毕] 数据湖灌注重构打平完成。你的结构已经满足脚本祖宗们的偏执要求。\n")
    return parent_cmd_dir

def run_command_safely(cmd_list, step_name):
    """ 安全地执行系统命令并阻塞式报错阻断法 """
    print(f"=============================================")
    print(f"⏳ 正在执行重型管道步骤 ---> [{step_name}]")
    print(f"💻 正在调用并分发至系统底层级的 Shell 为: {' '.join(cmd_list)}")
    print(f"=============================================")
    try:
        # check=True 使得任何进程内只要不是以完美 exit 0 回归，都会抛异常炸毁！杜绝假象
        subprocess.run(cmd_list, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [管道中断] 在执行 '{step_name}' 这个管道任务时发生了严重崩溃中断！")
        print(f"❌ [崩溃现场] 系统的错误退出退出码为：{e.returncode}")
        print("❌ 根据架构防误导预设，已经将后续所有一切流程彻底掐断销毁！")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GFlow Data Automation Setup -- 拒绝人肉填海的安全一键预处理全家桶")
    parser.add_argument("--input", "-i", type=str, required=True, help="哪怕是个混杂文件的文件夹，只要包括图像序列就会被抓出识别的目标源。")
    parser.add_argument("--target", "-t", type=str, default="./dataset", help="我们要强行生成标准三层结构的总控基石目录（默认会放到 ./dataset）。")
    parser.add_argument("--name", "-n", type=str, default=None, help="重塑这个场次时的 Scene 场景主名，不填就默认为你抓包输入文件名字。")
    parser.add_argument("--shutdown", action="store_true", help="加入此命令可在所有的流程大满贯成功结束之后替你关服落灯。")
    
    args = parser.parse_args()
    
    # 解析需要统称的场景名
    scene_name = args.name if args.name else Path(args.input).name
    if not scene_name:
        scene_name = "default_scene_unnamed"
        
    # Phase 1: 解析你给定的混杂原位，做成强迫症式的严密文件系统给后面的祖宗代码用
    pipeline_target_dir = create_and_organize_dataset(args.input, args.target, scene_name)
    target_str = str(pipeline_target_dir) # 获取如 "./dataset/Truck" 给 bash 使用
    
    # Phase 2: 管线链排队组合配置
    # 如果其中某个要新加参数配置也可以单独改这里
    pipeline_steps = [
        (["bash", "./scripts/depth_mast3r.sh", target_str], "1. Mast3r 三维相对深度估值与重建提取"),
        (["bash", "./scripts/flow_unimatch.sh", target_str], "2. Unimatch 强鲁棒性光流全局计算追踪"),
        (["bash", "./scripts/move_seg.sh", target_str], "3. 高精度时序运动分析与实例分割扣取生成")
    ]
    
    for cmd, desc in pipeline_steps:
        run_command_safely(cmd, desc)
        
    print("\n✅ [任务大满贯] 所有的工作管道都已完整通过零容忍生死考验！你的数据准备完成。")
    
    if args.shutdown:
        print("⚠️ 接收到你的指令，由于工作完成，服务器即将通过底层权限拉闸关机！")
        # 关机时不抛错避免引发未知阻塞异常
        subprocess.run(["/usr/bin/shutdown"], check=False)

if __name__ == "__main__":
    main()
