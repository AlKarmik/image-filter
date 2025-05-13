# image-filter
Image Filter for Content Tagging
This project provides a Python script (fast_filter_minors.py) to filter images based on specific tags using the wdtagger library (v0.14.0, swinv2-tagger-v3 model). It is optimized for Windows 10 with an NVIDIA GPU (e.g., RTX 3090) and can process ~200,000 images in ~2.2 hours. Images are sorted into "safe" (F:\adult_only) and "flagged" (F:\photos\_minors_flagged) folders based on a tag list (minors.txt).
Features

Fast Processing: Uses GPU acceleration (CUDA) for tagging ~128 images in ~4.2 seconds.
Custom Tags: Filters images with tags like loli, shota, child, etc., listed in minors.txt.
Robust: Skips corrupted files, handles duplicates, and logs all actions (filter_log.txt).
Portable: Includes setup scripts (setup.bat, run.bat) for easy deployment.

Requirements

OS: Windows 10/11 (64-bit).
GPU: NVIDIA GPU with CUDA support (e.g., RTX 3090, ≥11.8 CUDA version).
Disk: ~100 GB free space for 200,000 images.
Internet: For downloading drivers, Anaconda, and packages.
Admin Rights: For installing drivers and software.

Installation
1. Install NVIDIA Drivers

Check current driver:nvidia-smi

Ensure Driver Version is recent (e.g., 576.40) and CUDA Version ≥11.8.
If outdated or missing, download the latest driver for your GPU from NVIDIA.
Install with "Clean Install" option, reboot.

2. Install Anaconda

Download Anaconda (Python 3.10+) for Windows from anaconda.com (~500 MB, 64-bit).
Run installer, select "Just Me", install to C:\Users\YourUser\Anaconda3.
Check "Add Anaconda to PATH" (ignore warning).
Verify:conda --version

Should output, e.g., conda 23.7.4.

3. Set Up Environment

Clone this repository:
git clone https://github.com/yourusername/image-filter.git
cd image-filter


Run setup.bat to create the wd14_env environment and install dependencies:
setup.bat

This:

Creates wd14_env with Python 3.10.
Installs dependencies from requirements.txt (PyTorch with CUDA, wdtagger, etc.).
Verifies CUDA availability.

Note: Ensure requirements.txt is in the repo root:
torch==2.7.0+cu118
torchvision==0.22.0+cu118
torchaudio==2.7.0+cu118
pillow
tqdm
wdtagger==0.14.0
onnxruntime-gpu==1.22.0

4. Prepare Files

Images: Place images (.jpg, .jpeg, .png, .webp) in F:\photos.
Output Folder: Ensure F:\adult_only is writable. F:\photos\_minors_flagged will be created automatically.
Tag List: Use minors.txt (included) or customize it.

Save as F:\minors.txt, one tag per line, UTF-8 encoding.

Usage

Activate environment:
conda activate wd14_env

Run the script:
python fast_filter_minors.py --src "F:\photos" --dst "F:\adult_only" --batch 128 --thr 0.3

Or use run.bat:
run.bat

Parameters:

--src: Source folder (F:\photos).
--dst: Safe images folder (F:\adult_only).
--batch: Batch size (128 for RTX 3090, reduce to 64 if VRAM issues).
--thr: Tag confidence threshold (0.3 for high sensitivity, increase to 0.4 to reduce false positives).
--bad: Tag list file (F:\minors.txt by default).

Performance

Time: ~ 3 hours for 200,000 images (1898 batches × ~5 sec/batch).
VRAM: ~8–12 GB for batch 128 on RTX 3090 (24 GB total).
Disk: Ensure ~100 GB free on F:\ for image sorting.

Troubleshooting

CUDA Not Available:
Check:python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Should output: True NVIDIA GeForce RTX 3090.
If False, reinstall PyTorch:pip uninstall torch torchvision torchaudio
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118




VRAM Overflow:
Reduce batch size:python fast_filter_minors.py --src "F:\photos" --dst "F:\adult_only" --batch 64 --thr 0.3




False Positives:
Remove broad tags (cute, blushing, moe) from minors.txt or increase --thr:python fast_filter_minors.py --src "F:\photos" --dst "F:\adult_only" --batch 128 --thr 0.4




Missed Images:
Test a missed image:python -c "from wdtagger import Tagger; from PIL import Image; tagger = Tagger(); img = Image.open('F:\\photos\\159124.jpg').convert('RGB'); results = tagger.tag([img])[0]; print({**results.general_tag_data, **results.character_tag_data})"


Add missing tags to minors.txt.


Corrupted Files:
Skipped files are logged in filter_log.txt.



Files

fast_filter_minors.py: Main script for tagging and sorting images.
minors.txt: List of tags to flag (e.g., loli, shota).
requirements.txt: Python dependencies.
setup.bat: Automates environment setup.
run.bat: Runs the script with default parameters.

License
MIT License (or choose your preferred license).
Contributing
Feel free to open issues or submit pull requests for improvements, additional tags, or alternative taggers (e.g., deepdanbooru).
Acknowledgments

Built with wdtagger (v0.14.0) and PyTorch (CUDA).
Optimized for NVIDIA RTX 3090 on Windows 10.

