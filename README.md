# 🎬 Shadow AI Video Replicator

**AI-powered video editing that copies any video's style automatically!**

Upload your original video + a reference video → AI analyzes both → Replicates the exact editing style with 100% code-based processing.

---

## ✨ How It Works

### 3-Step Process:

1. **📹 Upload** - Your original video + reference video
2. **🔍 Analyze** - AI detects: scene cuts, transitions, color grading, text overlays, zoom effects, audio levels
3. **✨ Replicate** - AI asks what changes you want (watermark, captions, etc.) then edits automatically

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Dual Video Analysis** | Analyzes both videos with OpenCV computer vision |
| **Scene Detection** | Auto-detects cuts and transitions |
| **Color Grading Copy** | Matches B&W, sepia, warm, cool tones |
| **Text Overlay Detection** | Finds text positions from reference |
| **Zoom Detection** | Detects zoom in/out effects |
| **Audio Matching** | Copies volume levels |
| **Pre-Edit Questions** | Asks for watermark, captions, custom notes |
| **JSON Storage** | No database - everything in JSON files |
| **Groq AI** | Optional AI-powered analysis summaries |
| **Mobile Design** | Professional dark theme, bottom nav |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Get Groq API Key for AI summaries
# https://console.groq.com - Free tier available

# 3. Run the app
python app.py

# 4. Open browser
http://localhost:5000
```

---

## 📝 What AI Detects

### From Reference Video:
- ✅ Scene cuts & transitions
- ✅ Fade in/out effects
- ✅ Color style (B&W, sepia, warm, cool)
- ✅ Brightness & contrast levels
- ✅ Text overlay positions
- ✅ Zoom in/out effects
- ✅ Audio volume levels
- ✅ Motion intensity

### What You Can Customize:
- 📝 **Watermark** - Add your watermark with position
- 💬 **Captions** - Add custom text overlays
- 🎨 **Color Match** - Toggle color grading copy
- ✨ **Transitions** - Toggle fade effects
- 📝 **Text Style** - Toggle text position matching
- 🔊 **Audio** - Toggle volume matching
- 📝 **Custom Notes** - Tell AI special requests

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Video Analysis**: OpenCV (Computer Vision)
- **Video Processing**: MoviePy + FFmpeg
- **AI Summaries**: Groq API (Llama 3.3-70B)
- **Storage**: JSON files (no database)
- **Frontend**: HTML5, CSS3, Vanilla JS

---

## 📁 Project Structure

```
shadow_ai_video_replicator/
├── app.py                    # Main Flask app
├── requirements.txt          # Dependencies
├── templates/
│   └── index.html           # Professional UI
├── uploads/                 # Video uploads
├── output/                  # Edited videos
├── temp_frames/             # Frame analysis
└── json_storage/            # JSON data storage
```

---

## 🔧 Requirements

### System Dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg imagemagick

# macOS
brew install ffmpeg imagemagick

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
```

### Python Dependencies:
All included in `requirements.txt`:
- Flask, Werkzeug
- MoviePy, imageio-ffmpeg
- OpenCV (cv2)
- NumPy, Pillow
- Groq (optional)

---

## 🎨 UI Features

- **3-Step Wizard** - Upload → Analyze → Edit
- **Animated Stepper** - Visual progress indicator
- **Drag & Drop** - Easy file uploads
- **Real-time Analysis** - Shows detected features
- **Toggle Switches** - Easy preference controls
- **Position Grid** - Visual watermark position selector
- **Progress Animation** - 5-step processing visualization
- **Error Toasts** - Beautiful error notifications
- **Bottom Navigation** - Mobile-optimized nav
- **Dark Theme** - Professional glassmorphism design

---

## 📱 Mobile Optimized

- Fixed max-width (480px)
- Bottom navigation bar
- Touch-friendly buttons
- No zoom/pinch
- Smooth scroll animations
- Responsive cards

---

## 🎯 Example Workflow

1. **Upload**: Your vlog video + trending TikTok video
2. **AI Detects**: "TikTok has fast cuts, warm colors, text at bottom, zoom effects"
3. **You Choose**: Add watermark "@YourChannel", keep color matching ON
4. **AI Edits**: Your vlog gets same fast cuts, warm colors, text positions, zoom effects + your watermark

---

## 🔑 Groq API Key (Optional)

For AI-powered analysis summaries:
1. Visit [console.groq.com](https://console.groq.com)
2. Create free account
3. Copy API key (starts with `gsk_`)
4. Paste in app Settings

**Works without API key too** - uses fallback analysis!

---

Made with ❤️ by Shadow
