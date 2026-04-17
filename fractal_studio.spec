# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Fractal Studio.

Build with:
    pyinstaller fractal_studio.spec
"""
import os

block_cipher = None
HERE = os.path.abspath('.')

a = Analysis(
    ['gui.py'],
    pathex=[HERE],
    binaries=[],
    datas=[
        # Include project modules that are imported at runtime
        ('generators', 'generators'),
        ('colorizers.py', '.'),
        ('mask.py', '.'),
        ('render.py', '.'),
        ('video.py', '.'),
        # Include any SVG masks shipped with the project
        ('*.svg', '.'),
    ],
    hiddenimports=[
        'generators',
        'generators._core',
        'generators.julia',
        'generators.flower',
        'colorizers',
        'mask',
        # qtconsole + IPython for the embedded terminal
        'qtconsole',
        'qtconsole.rich_jupyter_widget',
        'qtconsole.inprocess',
        'ipykernel',
        'ipykernel.inprocess',
        'ipykernel.inprocess.ipkernel',
        'ipython_pygments_lexers',
        # imageio backend for video encoding
        'imageio',
        'imageio.v3',
        'imageio.plugins',
        # matplotlib colormaps used by twilight colorizer
        'matplotlib',
        'matplotlib.colormaps',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FractalStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,         # Needed for --run-script subprocess mode
    icon=None,            # Add an .ico here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FractalStudio',
)

