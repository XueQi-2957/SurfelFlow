# GFlow 全自动预处理工具 (auto_preprocess.py) 使用指南

为了解决 GFlow 数据处理中繁琐的目录层级重组（三层嵌套限制）以及多步骤脚本手动调用的痛点，我们引入了根目录下的 `auto_preprocess.py`。该工具能够一键完成图片重组并顺序调用深度、光流、分割预处理管道。

## ✨ 核心亮点

1.  **自动三层同名目录重塑**：
    GFlow 要求数据集遵循 `./dataset/Scene/Scene/Scene/*.jpg` 这种极其严苛的三层同名嵌套。使用该脚本，你只需要传入一个包含图片的普通文件夹，它会自动在目标位置为你完成 **「原地剪切转移 (Move)」** 和目录构建。
2.  **铁腕管道执行 (Fail-fast)**：
    脚本采用了严格的任务链管理。`Mast3r (深度)` -> `Unimatch (光流)` -> `Segment (分割)`。如果其中任何一个环节（如显存溢出 OOM）挂了，主脚本会立即拦截报错并强行停止，绝不生成掩耳盗铃的错误结果。
3.  **环境兼容性加固**：
    自动补齐了 Shell 脚本缺失的 Shebang，并强制显式调用 Bash 和当前 Python 解释器，彻底告别 `OSError: Exec format error` 或 `ModuleNotFoundError`。

---

## 🚀 快速开始

在项目 **根目录** (`GFlow/`) 下运行：

### 1. 最简运行（原地处理）
假设你的图片就在 `./dataset/Nvidia-Jumping/` 下：
```bash
python auto_preprocess.py --input ./dataset/Nvidia-Jumping
```
*   **结果**：图片会被移动到 `./dataset/Nvidia-Jumping/Nvidia-Jumping/Nvidia-Jumping/`，然后自动开始跑深度和光流。*

### 2. 指定场景名
如果你想给这个序列起个正式点的名字（比如 `MyTest`）：
```bash
python auto_preprocess.py --input ./my_raw_images/ --name MyTest
```

### 3. 处理完自动关机 (适合深夜炼丹)
```bash
python auto_preprocess.py --input ./dataset/MyData --shutdown
```

---

## 🛠 参数详解

*   `--input` (`-i`): **必需**。输入图片所在的文件夹路径。
*   `--name` (`-n`): **可选**。指定场景名，默认使用输入文件夹的名字。
*   `--target` (`-t`): **可选**。数据集存放的基目录，默认为 `./dataset`。
*   `--shutdown`: **可选**。全部步骤成功跑完后，执行系统关机命令。

## ⚠️ 注意事项
*   **原地移动**：该脚本默认使用 `move` 逻辑以节省磁盘空间和 IO 开销。运行前请确认你已经对原始图片做了备份（如果需要的话）。
*   **环境要求**：请确保在 `conda activate mast3r` 环境下运行，以保证所有子模块的依赖库正常加载。
