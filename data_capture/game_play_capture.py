import ctypes
import win32gui
import win32ui
import win32con
from time import sleep


# This script is used for capturing game frames every 5 sections.
# Captured frames will be manually labeled for YOLO training.

frame_number = 0
sleep(1)

# Gets the game window location
ctypes.windll.shcore.SetProcessDpiAwareness(2)
hwnd = win32gui.FindWindow(None, "Counter-Strike: Global Offensive")
win32gui.SetForegroundWindow(hwnd)
#hwnd = win32gui.GetDesktopWindow()
l,t,r,b=win32gui.GetWindowRect(hwnd)
print(l,t,r,b)
h=b-t
w=r-l

while True:
	print(frame_number)
	sleep(5)
	hDC = win32gui.GetWindowDC(hwnd)
	myDC=win32ui.CreateDCFromHandle(hDC)
	newDC=myDC.CreateCompatibleDC()

	myBitMap = win32ui.CreateBitmap()
	myBitMap.CreateCompatibleBitmap(myDC, w, h)

	newDC.SelectObject(myBitMap)

	# Lame way to allow screen to draw before taking shot
	sleep(.2)
	newDC.BitBlt((0,0),(w, h) , myDC, (0,0), win32con.SRCCOPY)
	myBitMap.Paint(newDC)
	fileName = 'f' + str(frame_number) + '.bmp'
	myBitMap.SaveBitmapFile(newDC, fileName)

	myDC.DeleteDC()
	newDC.DeleteDC()
	win32gui.ReleaseDC(hwnd, hDC)
	win32gui.DeleteObject(myBitMap.GetHandle())
	i += 1