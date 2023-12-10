import pygetwindow as gw
import win32con
import win32gui

def set_foreground(window_title):
    try:
        # Get the window containing window_title
        window = gw.getWindowsWithTitle(window_title)[0]
        hwnd = window._hWnd
        # set window to front
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, window.left, window.top, window.width, window.height, 0)
    except IndexError:
        print(f"No window with title '{window_title}' was found.")
    except Exception as e:
        print(f"Error: {e}")

# Specify the PowerShell window title (partial match)
set_foreground("PowerShell")
