import os
import json
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify, session
from werkzeug.utils import secure_filename
import moviepy.editor as mp
from moviepy.video.fx.all import fadein, fadeout
from groq import Groq

app = Flask(__name__)
app.secret_key = 'shadow_video_replicator_2026_ultra_secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['TEMP_FRAMES'] = 'temp_frames'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
app.config['JSON_STORAGE'] = 'json_storage'

for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 
               app.config['TEMP_FRAMES'], app.config['JSON_STORAGE']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}

# ========== JSON STORAGE SYSTEM ==========
class JSONStorage:
    def __init__(self, base_path):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _get_path(self, key):
        return os.path.join(self.base_path, f"{key}.json")

    def save(self, key, data):
        with open(self._get_path(key), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, key, default=None):
        path = self._get_path(key)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default

    def delete(self, key):
        path = self._get_path(key)
        if os.path.exists(path):
            os.remove(path)

    def list_all(self):
        files = []
        for f in os.listdir(self.base_path):
            if f.endswith('.json'):
                files.append(f[:-5])
        return files

storage = JSONStorage(app.config['JSON_STORAGE'])

# ========== GROQ API ==========
def get_groq_client():
    api_key = session.get('groq_api_key', '')
    if api_key and api_key.startswith('gsk_'):
        return Groq(api_key=api_key)
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO

# ========== VIDEO ANALYSIS ENGINE ==========
class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def analyze_all(self):
        print(f"Analyzing: {self.video_path} | Duration: {self.duration:.2f}s | {self.width}x{self.height}")

        analysis = {
            'basic_info': {
                'duration': round(self.duration, 2),
                'fps': round(self.fps, 2),
                'width': self.width,
                'height': self.height,
                'total_frames': self.total_frames,
                'aspect_ratio': round(self.width / self.height, 2) if self.height > 0 else 0
            },
            'scene_changes': self.detect_scene_changes(),
            'color_analysis': self.analyze_colors(),
            'transitions': self.detect_transitions(),
            'text_regions': self.detect_text_regions(),
            'brightness_timeline': self.analyze_brightness_timeline(),
            'audio_analysis': self.analyze_audio(),
            'zoom_levels': self.detect_zoom_levels(),
            'motion_analysis': self.analyze_motion()
        }

        self.cap.release()
        return analysis

    def detect_scene_changes(self, threshold=30.0):
        scene_changes = []
        prev_frame = None
        frame_count = 0
        sample_interval = max(1, int(self.fps / 2))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame_count > self.total_frames:
                break

            if frame_count % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    if mean_diff > threshold:
                        scene_changes.append({
                            'timestamp': round(frame_count / self.fps, 2),
                            'frame': frame_count,
                            'intensity': round(mean_diff, 2)
                        })
                prev_frame = gray
            frame_count += 1

        return scene_changes

    def analyze_colors(self, num_samples=10):
        colors = []
        frame_step = max(1, self.total_frames // num_samples)

        for i in range(0, self.total_frames, frame_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            avg_color = np.mean(rgb, axis=(0, 1))
            r, g, b = avg_color
            std_rgb = np.std([r, g, b])
            is_bw = std_rgb < 15
            is_sepia = (r > g > b) and (r - b > 30) and not is_bw
            warmth = r / (b + 1) if b > 0 else 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            colors.append({
                'timestamp': round(i / self.fps, 2),
                'avg_rgb': [round(x, 1) for x in avg_color],
                'is_black_white': bool(is_bw),
                'is_sepia': bool(is_sepia),
                'warmth': round(warmth, 2),
                'brightness': round(float(brightness), 1)
            })

        return colors

    def detect_transitions(self):
        transitions = []
        brightness_data = self.analyze_brightness_timeline(50)

        if len(brightness_data) < 3:
            return transitions

        for i in range(1, min(5, len(brightness_data))):
            if brightness_data[i]['normalized'] > brightness_data[i-1]['normalized'] + 0.2:
                if brightness_data[0]['normalized'] < 0.2:
                    transitions.append({'type': 'fade_in', 'timestamp': brightness_data[i]['timestamp'], 'confidence': 'high'})
                    break

        for i in range(len(brightness_data)-1, max(len(brightness_data)-6, 0), -1):
            if brightness_data[i]['normalized'] < brightness_data[i-1]['normalized'] - 0.2:
                if brightness_data[-1]['normalized'] < 0.2:
                    transitions.append({'type': 'fade_out', 'timestamp': brightness_data[i]['timestamp'], 'confidence': 'high'})
                    break

        return transitions

    def detect_text_regions(self, num_samples=5):
        text_regions = []
        frame_step = max(1, self.total_frames // num_samples)

        for i in range(0, self.total_frames, frame_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                if 2 < aspect_ratio < 15 and 10 < h < 100 and w > 50:
                    regions.append({'x': x, 'y': y, 'w': w, 'h': h, 'position': self._get_position(x, y, w, h)})

            if regions:
                text_regions.append({'timestamp': round(i / self.fps, 2), 'regions': regions[:5]})

        return text_regions

    def _get_position(self, x, y, w, h):
        cx = x + w / 2
        cy = y + h / 2
        if cy < self.height * 0.3:
            return 'top'
        elif cy > self.height * 0.7:
            return 'bottom'
        elif cx < self.width * 0.3:
            return 'left'
        elif cx > self.width * 0.7:
            return 'right'
        return 'center'

    def analyze_brightness_timeline(self, num_samples=20):
        brightness = []
        frame_step = max(1, self.total_frames // num_samples)

        for i in range(0, self.total_frames, frame_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            brightness.append({
                'timestamp': round(i / self.fps, 2),
                'brightness': round(float(avg_brightness), 1),
                'normalized': round(float(avg_brightness / 255), 2)
            })

        return brightness

    def analyze_audio(self):
        try:
            clip = mp.VideoFileClip(self.video_path)
            if clip.audio is None:
                return {'has_audio': False}

            audio = clip.audio
            duration = audio.duration
            volumes = []

            for t in np.linspace(0, duration, min(20, int(duration) + 1)):
                if t < duration:
                    try:
                        frame = audio.get_frame(t)
                        if isinstance(frame, np.ndarray):
                            volume = np.sqrt(np.mean(frame**2))
                            volumes.append(round(float(volume), 3))
                    except:
                        pass

            clip.close()
            return {
                'has_audio': True,
                'duration': round(duration, 2),
                'avg_volume': round(np.mean(volumes), 3) if volumes else 0,
                'max_volume': round(max(volumes), 3) if volumes else 0
            }
        except Exception as e:
            return {'has_audio': False, 'error': str(e)}

    def detect_zoom_levels(self, num_samples=10):
        zooms = []
        frame_step = max(1, self.total_frames // num_samples)
        prev_edges = None

        for i in range(0, self.total_frames, frame_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_count = np.sum(edges > 0)

            if prev_edges is not None and prev_edges > 0:
                ratio = edge_count / prev_edges
                if ratio > 1.3:
                    zooms.append({'timestamp': round(i/self.fps, 2), 'type': 'zoom_in', 'ratio': round(ratio, 2)})
                elif ratio < 0.7:
                    zooms.append({'timestamp': round(i/self.fps, 2), 'type': 'zoom_out', 'ratio': round(ratio, 2)})

            prev_edges = edge_count if edge_count > 0 else 1

        return zooms

    def analyze_motion(self):
        motions = []
        prev_frame = None
        frame_count = 0
        sample_interval = max(1, int(self.fps))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame_count > self.total_frames:
                break

            if frame_count % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
                    mean_motion = np.mean(magnitude)
                    motions.append({
                        'timestamp': round(frame_count / self.fps, 2),
                        'motion_intensity': round(float(mean_motion), 2),
                        'is_static': bool(mean_motion < 1.0)
                    })
                prev_frame = gray
            frame_count += 1

        return motions


# ========== AI EDITING PLAN GENERATOR ==========
def generate_editing_plan(original_analysis, reference_analysis, user_preferences):
    plan = {
        'metadata': {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_duration': original_analysis['basic_info']['duration'],
            'reference_duration': reference_analysis['basic_info']['duration'],
            'user_preferences': user_preferences
        },
        'structure_plan': [],
        'effects_plan': [],
        'text_overlays': [],
        'audio_plan': {},
        'color_grading': {}
    }

    orig_duration = original_analysis['basic_info']['duration']
    ref_duration = reference_analysis['basic_info']['duration']
    scale_factor = orig_duration / ref_duration if ref_duration > 0 else 1

    # 1. Scene structure matching
    ref_scenes = reference_analysis['scene_changes']
    if ref_scenes:
        for scene in ref_scenes:
            mapped_time = scene['timestamp'] * scale_factor
            plan['structure_plan'].append({
                'type': 'scene_cut',
                'reference_time': scene['timestamp'],
                'original_time': round(mapped_time, 2),
                'intensity': scene['intensity']
            })

    # 2. Color grading
    ref_colors = reference_analysis.get('color_analysis', [])
    if ref_colors:
        is_bw = any(c['is_black_white'] for c in ref_colors)
        is_sepia = any(c['is_sepia'] for c in ref_colors)

        if is_bw:
            plan['color_grading']['grayscale'] = True
        elif is_sepia:
            plan['color_grading']['sepia'] = True
        else:
            avg_brightness = np.mean([c['brightness'] for c in ref_colors])
            orig_brightness = np.mean([c['brightness'] for c in original_analysis.get('color_analysis', [])]) if original_analysis.get('color_analysis') else 128
            brightness_factor = avg_brightness / (orig_brightness + 1)
            plan['color_grading']['brightness'] = round(min(brightness_factor, 2.0), 2)
            plan['color_grading']['contrast'] = 1.2

            avg_warmth = np.mean([c['warmth'] for c in ref_colors])
            if avg_warmth > 1.2:
                plan['color_grading']['warmth'] = True
            elif avg_warmth < 0.8:
                plan['color_grading']['cool'] = True

    # 3. Transitions
    ref_transitions = reference_analysis.get('transitions', [])
    for trans in ref_transitions:
        mapped_time = trans['timestamp'] * scale_factor
        plan['effects_plan'].append({'type': trans['type'], 'timestamp': round(mapped_time, 2), 'duration': 2.0})

    # 4. Zoom effects
    ref_zooms = reference_analysis.get('zoom_levels', [])
    for zoom in ref_zooms[:2]:
        mapped_time = zoom['timestamp'] * scale_factor
        plan['effects_plan'].append({'type': zoom['type'], 'timestamp': round(mapped_time, 2), 'value': 1.5 if zoom['type'] == 'zoom_in' else 0.7})

    # 5. Text overlays from reference
    ref_text = reference_analysis.get('text_regions', [])
    if ref_text and user_preferences.get('add_captions', True):
        for text_region in ref_text[:3]:
            positions = [r['position'] for r in text_region.get('regions', [])]
            most_common = max(set(positions), key=positions.count) if positions else 'center'
            plan['text_overlays'].append({
                'text': user_preferences.get('caption_text', 'Shadow Edit'),
                'position': most_common,
                'start_time': round(text_region['timestamp'], 2),
                'duration': 5.0,
                'fontsize': 50,
                'color': 'white'
            })

    # 6. Audio
    ref_audio = reference_analysis.get('audio_analysis', {})
    orig_audio = original_analysis.get('audio_analysis', {})
    if ref_audio.get('has_audio') and orig_audio.get('has_audio'):
        volume_ratio = ref_audio.get('avg_volume', 1) / (orig_audio.get('avg_volume', 1) + 0.001)
        plan['audio_plan']['volume_adjust'] = round(min(volume_ratio, 2.0), 2)

    # 7. Watermark
    if user_preferences.get('watermark_text'):
        plan['text_overlays'].append({
            'text': user_preferences['watermark_text'],
            'position': user_preferences.get('watermark_position', 'bottom-right'),
            'start_time': 0,
            'duration': orig_duration,
            'fontsize': 20,
            'color': 'white',
            'opacity': 0.5,
            'is_watermark': True
        })

    return plan


# ========== VIDEO PROCESSOR ==========
def apply_editing_plan(video_path, plan, output_path):
    clip = mp.VideoFileClip(video_path)

    # Color grading
    color = plan.get('color_grading', {})
    if color.get('grayscale'):
        clip = clip.fx(mp.vfx.blackwhite)

    if color.get('sepia'):
        def sepia_frame(frame):
            r = frame[:,:,0] * 0.393 + frame[:,:,1] * 0.769 + frame[:,:,2] * 0.189
            g = frame[:,:,0] * 0.349 + frame[:,:,1] * 0.686 + frame[:,:,2] * 0.168
            b = frame[:,:,0] * 0.272 + frame[:,:,1] * 0.534 + frame[:,:,2] * 0.131
            return np.stack([r, g, b], axis=2).clip(0, 255).astype(np.uint8)
        clip = clip.fl_image(sepia_frame)

    if 'brightness' in color:
        clip = clip.fx(mp.vfx.colorx, color['brightness'])

    if 'contrast' in color:
        contrast_val = color['contrast']
        def contrast_frame(frame):
            return np.clip((frame.astype(float) - 128) * contrast_val + 128, 0, 255).astype(np.uint8)
        clip = clip.fl_image(contrast_frame)

    # Effects
    effects = plan.get('effects_plan', [])

    # Speed
    speed_effects = [e for e in effects if e['type'] == 'speed']
    if speed_effects:
        avg_speed = np.mean([e.get('value', 1.0) for e in speed_effects])
        if avg_speed != 1.0:
            clip = clip.fx(mp.vfx.speedx, avg_speed)

    # Fades
    for fade in effects:
        if fade['type'] == 'fade_in':
            clip = clip.fx(fadein, duration=fade.get('duration', 2))
        elif fade['type'] == 'fade_out':
            clip = clip.fx(fadeout, duration=fade.get('duration', 2))

    # Zoom
    for zoom in effects:
        if zoom['type'] == 'zoom_in':
            def zoom_in_frame(get_frame, t):
                frame = get_frame(t)
                h, w = frame.shape[:2]
                progress = t / clip.duration if clip.duration > 0 else 0
                zoom_factor = 1 + (zoom.get('value', 1.5) - 1) * progress
                new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
                y1, x1 = (h - new_h) // 2, (w - new_w) // 2
                cropped = frame[y1:y1+new_h, x1:x1+new_w]
                return cv2.resize(cropped, (w, h))
            clip = clip.fl(zoom_in_frame)
        elif zoom['type'] == 'zoom_out':
            def zoom_out_frame(get_frame, t):
                frame = get_frame(t)
                h, w = frame.shape[:2]
                progress = t / clip.duration if clip.duration > 0 else 0
                zoom_factor = zoom.get('value', 0.7) + (1 - zoom.get('value', 0.7)) * progress
                new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
                resized = cv2.resize(frame, (new_w, new_h))
                result = np.zeros_like(frame)
                y1, x1 = (h - new_h) // 2, (w - new_w) // 2
                result[y1:y1+new_h, x1:x1+new_w] = resized
                return result
            clip = clip.fl(zoom_out_frame)

    # Text overlays
    text_overlays = plan.get('text_overlays', [])
    txt_clips = []

    for text_data in text_overlays:
        txt = text_data['text']
        pos = text_data.get('position', 'center')
        fontsize = text_data.get('fontsize', 50)
        color = text_data.get('color', 'white')
        start = text_data.get('start_time', 0)
        duration = text_data.get('duration', 5)
        opacity = text_data.get('opacity', 1.0)

        try:
            txt_clip = mp.TextClip(
                txt, fontsize=fontsize, color=color, font='Arial-Bold',
                stroke_color='black', stroke_width=2, method='caption',
                size=(clip.w * 0.8, None)
            )
        except Exception as e:
            print(f"TextClip error (likely ImageMagick missing): {e}")
            continue

        if pos == 'top':
            txt_clip = txt_clip.set_position(('center', 50))
        elif pos == 'bottom':
            txt_clip = txt_clip.set_position(('center', clip.h - 100))
        elif pos == 'bottom-right':
            txt_clip = txt_clip.set_position((clip.w - 200, clip.h - 60))
        elif pos == 'left':
            txt_clip = txt_clip.set_position((50, 'center'))
        elif pos == 'right':
            txt_clip = txt_clip.set_position((clip.w - 200, 'center'))
        else:
            txt_clip = txt_clip.set_position('center')

        txt_clip = txt_clip.set_start(start).set_duration(min(duration, clip.duration - start))
        if opacity < 1.0:
            txt_clip = txt_clip.set_opacity(opacity)
        txt_clips.append(txt_clip)

    if txt_clips:
        clip = mp.CompositeVideoClip([clip] + txt_clips)

    # Audio
    audio_plan = plan.get('audio_plan', {})
    if 'volume_adjust' in audio_plan and clip.audio is not None:
        clip = clip.volumex(audio_plan['volume_adjust'])

    # Write output
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=30, preset='medium', threads=4, logger=None)
    clip.close()
    return output_path


# ========== AI SUMMARY ==========
def detect_color_style(analysis):
    colors = analysis.get('color_analysis', [])
    if not colors:
        return 'normal'
    bw_count = sum(1 for c in colors if c.get('is_black_white'))
    sepia_count = sum(1 for c in colors if c.get('is_sepia'))
    if bw_count > len(colors) * 0.5:
        return 'black_white'
    if sepia_count > len(colors) * 0.3:
        return 'sepia'
    avg_warmth = np.mean([c.get('warmth', 1) for c in colors])
    if avg_warmth > 1.3:
        return 'warm'
    if avg_warmth < 0.8:
        return 'cool'
    return 'normal'


def generate_ai_summary(orig_analysis, ref_analysis):
    client = get_groq_client()
    if not client:
        ref_scenes = len(ref_analysis.get('scene_changes', []))
        ref_trans = len(ref_analysis.get('transitions', []))
        ref_text = len(ref_analysis.get('text_regions', []))
        return {
            'title': 'Video Analysis Complete',
            'description': f'Reference video has {ref_scenes} scene cuts, {ref_trans} transitions, and {ref_text} text regions detected.',
            'detected_style': detect_color_style(ref_analysis),
            'suggestions': ['Add matching scene cuts', 'Apply similar color grading', 'Add text overlays']
        }

    try:
        prompt = f"""Analyze these video analyses and provide summary JSON:

Reference Video:
- Duration: {ref_analysis['basic_info']['duration']}s
- Scene Changes: {len(ref_analysis.get('scene_changes', []))}
- Transitions: {ref_analysis.get('transitions', [])}
- Color Style: {detect_color_style(ref_analysis)}
- Text Regions: {len(ref_analysis.get('text_regions', []))}
- Audio: {'Yes' if ref_analysis.get('audio_analysis', {}).get('has_audio') else 'No'}

Original Video:
- Duration: {orig_analysis['basic_info']['duration']}s

Return JSON: {{"title": "", "description": "", "detected_style": "", "key_features": [], "suggestions": []}}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        return json.loads(result.strip())
    except:
        return {
            'title': 'Analysis Complete',
            'description': 'Reference video analyzed successfully',
            'detected_style': detect_color_style(ref_analysis),
            'key_features': ['Scene cuts detected', 'Color profile analyzed'],
            'suggestions': ['Apply matching edits', 'Add text overlays']
        }


# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    if api_key and api_key.startswith('gsk_'):
        session['groq_api_key'] = api_key
        return jsonify({'success': True, 'message': 'API Key saved!'})
    return jsonify({'success': False, 'error': 'Invalid API Key'})

@app.route('/analyze', methods=['POST'])
def analyze_videos():
    if 'original_video' not in request.files or 'reference_video' not in request.files:
        return jsonify({'error': 'Both videos required'}), 400

    original = request.files['original_video']
    reference = request.files['reference_video']

    if not original.filename or not reference.filename:
        return jsonify({'error': 'Both files must be selected'}), 400

    if not allowed_file(original.filename) or not allowed_file(reference.filename):
        return jsonify({'error': 'Invalid video format'}), 400

    timestamp = int(time.time())
    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], f"orig_{timestamp}_{secure_filename(original.filename)}")
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{timestamp}_{secure_filename(reference.filename)}")

    original.save(orig_path)
    reference.save(ref_path)

    try:
        print("Analyzing original video...")
        orig_analyzer = VideoAnalyzer(orig_path)
        orig_analysis = orig_analyzer.analyze_all()

        print("Analyzing reference video...")
        ref_analyzer = VideoAnalyzer(ref_path)
        ref_analysis = ref_analyzer.analyze_all()

        analysis_id = f"analysis_{timestamp}"
        storage.save(analysis_id, {
            'original_path': orig_path,
            'reference_path': ref_path,
            'original_analysis': orig_analysis,
            'reference_analysis': ref_analysis,
            'timestamp': timestamp
        })

        summary = generate_ai_summary(orig_analysis, ref_analysis)

        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'original_info': orig_analysis['basic_info'],
            'reference_info': ref_analysis['basic_info'],
            'summary': summary,
            'detected_effects': {
                'reference': {
                    'scene_cuts': len(ref_analysis.get('scene_changes', [])),
                    'transitions': len(ref_analysis.get('transitions', [])),
                    'text_regions': len(ref_analysis.get('text_regions', [])),
                    'color_style': detect_color_style(ref_analysis),
                    'has_audio': ref_analysis.get('audio_analysis', {}).get('has_audio', False)
                }
            }
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/preferences', methods=['POST'])
def save_preferences():
    data = request.get_json()
    analysis_id = data.get('analysis_id')
    if not analysis_id:
        return jsonify({'error': 'Analysis ID required'}), 400

    analysis_data = storage.load(analysis_id)
    if not analysis_data:
        return jsonify({'error': 'Analysis not found'}), 404

    preferences = {
        'add_captions': data.get('add_captions', True),
        'caption_text': data.get('caption_text', ''),
        'watermark_text': data.get('watermark_text', ''),
        'watermark_position': data.get('watermark_position', 'bottom-right'),
        'color_match': data.get('color_match', True),
        'speed_match': data.get('speed_match', True),
        'transition_match': data.get('transition_match', True),
        'text_style_match': data.get('text_style_match', True),
        'audio_match': data.get('audio_match', True),
        'custom_notes': data.get('custom_notes', '')
    }

    analysis_data['preferences'] = preferences
    plan = generate_editing_plan(analysis_data['original_analysis'], analysis_data['reference_analysis'], preferences)
    analysis_data['editing_plan'] = plan
    storage.save(analysis_id, analysis_data)

    return jsonify({
        'success': True,
        'plan': plan,
        'effects_count': len(plan.get('effects_plan', [])),
        'text_overlays_count': len(plan.get('text_overlays', [])),
        'color_grading': plan.get('color_grading', {})
    })

@app.route('/edit', methods=['POST'])
def edit_video():
    data = request.get_json()
    analysis_id = data.get('analysis_id')
    if not analysis_id:
        return jsonify({'error': 'Analysis ID required'}), 400

    analysis_data = storage.load(analysis_id)
    if not analysis_data:
        return jsonify({'error': 'Analysis not found'}), 404

    plan = analysis_data.get('editing_plan')
    if not plan:
        return jsonify({'error': 'No editing plan found'}), 400

    try:
        orig_path = analysis_data['original_path']
        timestamp = analysis_data['timestamp']
        output_filename = f"edited_{timestamp}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        apply_editing_plan(orig_path, plan, output_path)

        analysis_data['output_path'] = output_path
        analysis_data['output_filename'] = output_filename
        analysis_data['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        storage.save(f"completed_{analysis_id}", analysis_data)

        return jsonify({
            'success': True,
            'download_url': f'/download/{output_filename}',
            'preview_url': f'/preview/{output_filename}',
            'plan_summary': {
                'effects_applied': len(plan.get('effects_plan', [])),
                'text_overlays': len(plan.get('text_overlays', [])),
                'color_grading': plan.get('color_grading', {}),
                'structure_cuts': len(plan.get('structure_plan', []))
            }
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/preview/<filename>')
def preview(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path)
    return jsonify({'error': 'File not found'}), 404

@app.route('/history')
def get_history():
    completed = [k for k in storage.list_all() if k.startswith('completed_')]
    history = []
    for key in completed[:10]:
        data = storage.load(key)
        if data:
            history.append({
                'id': key,
                'timestamp': data.get('completed_at', ''),
                'original_name': os.path.basename(data.get('original_path', '')),
                'output_filename': data.get('output_filename', ''),
                'effects_count': len(data.get('editing_plan', {}).get('effects_plan', []))
            })
    return jsonify({'history': history})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
