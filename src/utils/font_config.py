"""
matplotlib 한글 폰트 설정 유틸리티
모든 분석 모듈에서 일관된 폰트 설정을 위한 공통 함수
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings

def setup_korean_font():
    """크로스 플랫폼 한글 폰트 설정 함수"""
    try:
        # 헤드리스 모드 설정
        if matplotlib.get_backend().lower() != 'agg':
            matplotlib.use('Agg', force=True)

        # 폰트 캐시 재구성 (조심스럽게)
        try:
            fm._rebuild()
        except:
            pass

        current_platform = platform.system()
        selected_font = None

        # 플랫폼별 한글 폰트 설정
        if current_platform == 'Windows':
            selected_font = _setup_windows_korean_font()
        elif current_platform == 'Linux':
            selected_font = _setup_linux_korean_font()
        elif current_platform == 'Darwin':  # macOS
            selected_font = _setup_macos_korean_font()

        # 공통 폰트 설정
        if selected_font:
            plt.rcParams.update({
                'font.family': [selected_font],
                'font.sans-serif': [selected_font, 'DejaVu Sans', 'Liberation Sans'],
                'axes.unicode_minus': False
            })
            print(f"선택된 폰트: {selected_font}")
        else:
            # 최후의 대안 - 기본 폰트로 폴백
            plt.rcParams.update({
                'font.family': ['DejaVu Sans'],
                'axes.unicode_minus': False
            })
            warnings.warn("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다. 한글이 정상 표시되지 않을 수 있습니다.")

    except Exception as e:
        warnings.warn(f"폰트 설정 중 오류 발생: {e}")
        plt.rcParams.update({
            'font.family': ['DejaVu Sans'],
            'axes.unicode_minus': False
        })

def _setup_windows_korean_font():
    """Windows 환경 한글 폰트 설정"""
    import os

    # Windows 한글 폰트 목록 (우선순위)
    windows_fonts = [
        'Malgun Gothic', 'Microsoft YaHei', 'SimHei',
        'MS Gothic', 'Gulim', 'Dotum', 'Batang'
    ]

    # 설치된 폰트에서 찾기
    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    for font in windows_fonts:
        if font in available_fonts:
            return font

    # 직접 시스템 폰트 경로에서 찾기
    try:
        fonts_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        font_files = ['malgun.ttf', 'malgunbd.ttf', 'gulim.ttc']

        for font_file in font_files:
            font_path = os.path.join(fonts_dir, font_file)
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                if font_file.startswith('malgun'):
                    return 'Malgun Gothic'
                elif font_file.startswith('gulim'):
                    return 'Gulim'
    except:
        pass

    return None

def _setup_linux_korean_font():
    """Linux/Ubuntu 환경 한글 폰트 설정"""
    import os
    import subprocess

    # Ubuntu/Linux 한글 폰트 목록 (우선순위)
    linux_fonts = [
        'Noto Sans CJK KR', 'Noto Sans KR', 'NanumGothic', 'NanumBarunGothic',
        'UnDotum', 'Baekmuk Dotum', 'WenQuanYi Micro Hei', 'Droid Sans Fallback'
    ]

    # 설치된 폰트에서 찾기
    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    for font in linux_fonts:
        if font in available_fonts:
            return font

    # 한글 폰트가 없는 경우 자동 설치 시도 (권장 사항만 출력)
    try:
        # fc-list로 한글 폰트 확인
        result = subprocess.run(['fc-list', ':lang=ko'], capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            # 사용 가능한 한글 폰트가 있으면 첫 번째 찾은 것 사용
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    font_path = line.split(':')[0].strip()
                    try:
                        fm.fontManager.addfont(font_path)
                        # 폰트 이름 추출 시도
                        font_prop = fm.FontProperties(fname=font_path)
                        font_name = font_prop.get_name()
                        if font_name and font_name != 'DejaVu Sans':
                            return font_name
                    except:
                        continue
    except:
        pass

    # 일반적인 Linux 폰트 경로에서 찾기
    linux_font_paths = [
        '/usr/share/fonts/truetype/noto/',
        '/usr/share/fonts/truetype/nanum/',
        '/usr/share/fonts/truetype/liberation/',
        '/usr/share/fonts/opentype/noto/',
        '/system/fonts/',  # Android 호환
    ]

    for font_dir in linux_font_paths:
        if os.path.exists(font_dir):
            try:
                for font_file in os.listdir(font_dir):
                    if any(keyword in font_file.lower() for keyword in ['noto', 'nanum', 'cjk', 'kr']):
                        font_path = os.path.join(font_dir, font_file)
                        if font_file.endswith(('.ttf', '.otf')):
                            try:
                                fm.fontManager.addfont(font_path)
                                font_prop = fm.FontProperties(fname=font_path)
                                return font_prop.get_name()
                            except:
                                continue
            except:
                continue

    return None

def _setup_macos_korean_font():
    """macOS 환경 한글 폰트 설정"""
    # macOS 한글 폰트 목록
    macos_fonts = [
        'Apple SD Gothic Neo', 'Nanum Gothic', 'AppleGothic',
        'Helvetica Neue', 'Arial Unicode MS'
    ]

    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    for font in macos_fonts:
        if font in available_fonts:
            return font

    return None

def get_matplotlib():
    """matplotlib와 seaborn을 임포트하고 폰트 설정을 적용한 후 반환"""
    setup_korean_font()
    import seaborn as sns
    return plt, sns