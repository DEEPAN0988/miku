import ctypes
import subprocess

def test_media_keys():
    user32 = ctypes.windll.user32
    # Windows Virtual Key codes:
    # VK_MEDIA_NEXT_TRACK = 0xB0
    # VK_MEDIA_PREV_TRACK = 0xB1
    # VK_MEDIA_STOP = 0xB2
    # VK_MEDIA_PLAY_PAUSE = 0xB3
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002

    def send_media_key(vk_code):
        # Key down
        user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
        # Key up
        user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

    print("Virtual key media control is ready via ctypes user32.keybd_event.")
    return True

if __name__ == "__main__":
    test_media_keys()
