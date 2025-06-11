import os
import io
import base64
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

class LineAnalyzer:
    def __init__(self, A=None, B=None, C=None, equation=None):
        if equation:
            self.A, self.B, self.C = self.parse_equation(equation)
        else:
            self.A, self.B, self.C = A, B, C
    
    def parse_equation(self, equation):
        eq = equation.replace(' ', '').replace('=', '-(') + ')'
        
        if 'y=' in eq:
            parts = eq.split('y=')
            rhs = parts[1].replace(')', '')  # Fixed: access index [1]
            
            if 'x' in rhs:
                if '*x' in rhs:
                    m_part = rhs.split('*x')[0]  # Fixed: access index [0]
                    rest = rhs.split('*x')[1] if len(rhs.split('*x')) > 1 else '0'
                else:
                    m_part = rhs.split('x')[0]  # Fixed: access index [0]
                    rest = rhs.split('x')[1] if len(rhs.split('x')) > 1 else '0'
                
                m = float(m_part) if m_part and m_part != '+' and m_part != '-' else (1 if m_part != '-' else -1)
                c = float(rest) if rest else 0
            else:
                m = 0
                c = float(rhs)
            
            return -m, 1, -c
        else:
            eq = eq.replace('-(', '-').replace(')', '')
            A, B, C = 0, 0, 0
            
            if 'x' in eq:
                x_pattern = r'([+-]?\d*\.?\d*)[\*]?x'
                x_match = re.search(x_pattern, eq)
                if x_match:
                    coeff = x_match.group(1)
                    if coeff in ['', '+']:
                        A = 1
                    elif coeff == '-':
                        A = -1
                    else:
                        A = float(coeff)
            
            if 'y' in eq:
                y_pattern = r'([+-]?\d*\.?\d*)[\*]?y'
                y_match = re.search(y_pattern, eq)
                if y_match:
                    coeff = y_match.group(1)
                    if coeff in ['', '+']:
                        B = 1
                    elif coeff == '-':
                        B = -1
                    else:
                        B = float(coeff)
            
            const_pattern = r'([+-]?\d+\.?\d*)(?![xy])'
            const_matches = re.findall(const_pattern, eq)
            if const_matches:
                C = sum(float(match) for match in const_matches)
            
            return A, B, C
    
    @property
    def slope(self):
        if self.B == 0:
            return float('inf')
        return -self.A / self.B
    
    @property
    def y_intercept(self):
        if self.B == 0:
            return None
        return -self.C / self.B
    
    @property
    def x_intercept(self):
        if self.A == 0:
            return None
        return -self.C / self.A
    
    def point_distance(self, point):
        x0, y0 = point
        return abs(self.A * x0 + self.B * y0 + self.C) / np.sqrt(self.A**2 + self.B**2)
    
    def line_distance(self, other_line):
        if abs(self.slope - other_line.slope) < 1e-10 or (self.B == 0 and other_line.B == 0):
            if self.A != 0 or self.B != 0:
                norm1 = np.sqrt(self.A**2 + self.B**2)
                norm2 = np.sqrt(other_line.A**2 + other_line.B**2)
                return abs(self.C/norm1 - other_line.C/norm2)
            return 0
        return 0
    
    def intersection_point(self, other_line):
        det = self.A * other_line.B - other_line.A * self.B
        if abs(det) < 1e-10:
            return None
        x = (self.B * other_line.C - other_line.B * self.C) / det
        y = (other_line.A * self.C - self.A * other_line.C) / det
        return (x, y)
    
    def is_parallel(self, other_line):
        if self.B == 0 and other_line.B == 0:
            return True
        if self.B == 0 or other_line.B == 0:
            return False
        return abs(self.slope - other_line.slope) < 1e-10
    
    def is_perpendicular(self, other_line):
        if self.B == 0 and other_line.A == 0:
            return True
        if self.A == 0 and other_line.B == 0:
            return True
        if self.B != 0 and other_line.B != 0:
            return abs(self.slope * other_line.slope + 1) < 1e-10
        return False

def create_plot(lines, points=None, title="Line Analysis"):
    try:
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        x = np.linspace(-10, 10, 1000)
        
        for i, line in enumerate(lines):
            if line.B != 0:
                y = (-line.A * x - line.C) / line.B
                mask = (y >= -10) & (y <= 10)
                plt.plot(x[mask], y[mask], color=colors[i % len(colors)], 
                        linewidth=2, label=f'Line {i+1}')
            else:
                x_vert = -line.C / line.A
                y_vert = np.linspace(-10, 10, 1000)
                plt.plot(np.full_like(y_vert, x_vert), y_vert, 
                        color=colors[i % len(colors)], linewidth=2, label=f'Line {i+1}')
        
        if points:
            for i, point in enumerate(points):
                plt.plot(point[0], point[1], 'ro', markersize=8, label=f'Point {i+1}')  # Fixed: point[0], point[1]
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    except Exception as e:
        print(f"Plot error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        
        operation = data.get('operation')
        
        if operation == 'point_line_distance':
            equation = data.get('equation')
            if not equation:
                return jsonify({'success': False, 'error': 'Please enter a line equation'})
            
            try:
                point_x = float(data.get('point_x', 0))
                point_y = float(data.get('point_y', 0))
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'Please enter valid point coordinates'})
            
            try:
                line = LineAnalyzer(equation=equation)
                distance = line.point_distance((point_x, point_y))
                
                plot_url = create_plot([line], [(point_x, point_y)], 
                                     f"Point to Line Distance: {distance:.3f}")
                
                return jsonify({
                    'success': True,
                    'result': f"Distance: {distance:.6f}",
                    'plot': plot_url,
                    'details': {
                        'slope': line.slope if line.slope != float('inf') else 'Undefined',
                        'y_intercept': line.y_intercept,
                        'x_intercept': line.x_intercept
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': f'Calculation error: {str(e)}'})
                
        elif operation == 'line_line_distance':
            eq1 = data.get('equation1')
            eq2 = data.get('equation2')
            
            if not eq1 or not eq2:
                return jsonify({'success': False, 'error': 'Please enter both line equations'})
            
            try:
                line1 = LineAnalyzer(equation=eq1)
                line2 = LineAnalyzer(equation=eq2)
                
                if line1.is_parallel(line2):
                    distance = line1.line_distance(line2)
                    result = f"Lines are parallel. Distance: {distance:.6f}"
                    title = f"Parallel Lines - Distance: {distance:.3f}"
                else:
                    intersection = line1.intersection_point(line2)
                    result = f"Lines intersect at: {intersection}"
                    title = "Intersecting Lines"
                
                plot_url = create_plot([line1, line2], title=title)
                
                return jsonify({
                    'success': True,
                    'result': result,
                    'plot': plot_url,
                    'parallel': line1.is_parallel(line2),
                    'perpendicular': line1.is_perpendicular(line2)
                })
            except Exception as e:
                return jsonify({'success': False, 'error': f'Calculation error: {str(e)}'})
                
        elif operation == 'point_point_distance':
            try:
                x1, y1 = float(data.get('x1')), float(data.get('y1'))
                x2, y2 = float(data.get('x2')), float(data.get('y2'))
                
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                plot_url = create_plot([], [(x1, y1), (x2, y2)], 
                                     f"Point to Point Distance: {distance:.3f}")
                
                return jsonify({
                    'success': True,
                    'result': f"Distance: {distance:.6f}",
                    'plot': plot_url
                })
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'Please enter valid coordinates'})
                
        elif operation == 'line_properties':
            equation = data.get('equation')
            if not equation:
                return jsonify({'success': False, 'error': 'Please enter a line equation'})
            
            try:
                line = LineAnalyzer(equation=equation)
                
                plot_url = create_plot([line], title="Line Properties")
                
                return jsonify({
                    'success': True,
                    'result': "Line properties calculated",
                    'plot': plot_url,
                    'properties': {
                        'general_form': f"{line.A:.3f}x + {line.B:.3f}y + {line.C:.3f} = 0",
                        'slope': line.slope if line.slope != float('inf') else 'Undefined',
                        'y_intercept': line.y_intercept,
                        'x_intercept': line.x_intercept
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': f'Calculation error: {str(e)}'})
        
        return jsonify({'success': False, 'error': 'Invalid operation'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
