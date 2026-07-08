import re
import sys

def strip_comments(code):
    lines = code.split('\n')
    out_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Keep section headers
        if stripped.startswith('// =') or stripped.startswith('// -'):
            out_lines.append(line)
            continue
            
        # Find // but ignore if inside string (heuristic: check if " exists before // and odd number of quotes)
        # Actually simpler: if line starts with //, just drop it.
        if stripped.startswith('//'):
            continue
            
        # Inline comments: split by // if not rtsp:// or http://
        if '//' in line and not '://' in line:
            parts = line.split('//')
            out_lines.append(parts[0].rstrip())
        else:
            out_lines.append(line)
            
    # Remove consecutive empty lines
    final_lines = []
    for line in out_lines:
        if line.strip() == '':
            if len(final_lines) > 0 and final_lines[-1].strip() == '':
                continue
        final_lines.append(line)
        
    return '\n'.join(final_lines)

with open('/home/duc/Desktop/Detect-RACK-Project/DetectRackProject/main.cpp', 'r', encoding='utf-8') as f:
    code = f.read()
    
new_code = strip_comments(code)
new_code = new_code.replace('YOLOv8Detector detector(modelPath, cv::Size(640, 640), 0.20f, 0.45f);', 'YOLOv8Detector detector(modelPath, cv::Size(640, 640), 0.65f, 0.45f);')

with open('/home/duc/Desktop/Detect-RACK-Project/DetectRackProject/main.cpp', 'w', encoding='utf-8') as f:
    f.write(new_code)
