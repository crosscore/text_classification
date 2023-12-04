import pygetwindow as gw
import win32con
import win32gui

def set_foreground(window_title):
    try:
        # window_titleが含まれるウィンドウを取得
        window = gw.getWindowsWithTitle(window_title)[0]
        hwnd = window._hWnd
        # ウィンドウを最前面に設定
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, window.left, window.top, window.width, window.height, 0)
    except IndexError:
        print(f"'{window_title}' タイトルのウィンドウが見つかりませんでした。")
    except Exception as e:
        print(f"エラー: {e}")

# PowerShellウィンドウのタイトルを指定（部分一致）
set_foreground("PowerShell")
