import cv2
import tkinter as tk
from PIL import Image, ImageTk
from depth_map import DepthMap


cap1 = cv2.VideoCapture(0)

def show_video(label1):
    while True:
        ret1, frame1 = cap1.read()
        frame1 = cv2.flip(frame1, 1)
        frame1 = cv2.resize(frame1, (480, 270))
        if not ret1:
            break
        img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        imgtk1 = ImageTk.PhotoImage(image=img1)
        label1.config(image=imgtk1)
        label1.image = imgtk1
        root.update()

    cap1.release()


def capture_photo(label2, label3):
    ret1, frame1 = cap1.read()
    frame1 = cv2.flip(frame1, 1)
    frame1 = cv2.resize(frame1, (480, 270))  # Изменение размера кадра
    cv2.imwrite('photo.jpg', frame1)
    print("Фото сделано")
    dm = DepthMap(label2)
    dm.get_map()
    width, height = dm.width, dm.height

    image = Image.open('photo.jpg')
    image = image.resize((width, height))
    image_tk = ImageTk.PhotoImage(image)
    label3.config(image=image_tk)
    label3.image = image_tk


root = tk.Tk()
root.title("Построение карты глубины")
root.minsize(1280, 740)
# root.maxsize(1290, 1200)
root.geometry("1280x760+75+0")
root.config(bg='#481a6c')

label1 = tk.Label(root, borderwidth=10, highlightcolor="red")
label1.place(x=400, y=20)

label2 = tk.Label(root,  bg="#1fa286")
label2.place(x=20, y=400)

label3 = tk.Label(root, bg="#1fa286")
label3.place(x=650, y=400)


button_shoot = tk.Button(root, bg='#395778', text='Snapshot', width=15, height=1, font=('Apple Chancery', 32, ''),
                         command=lambda: capture_photo(label2, label3), pady=3, padx=0, highlightbackground="#DDD9ED")
button_shoot.place(x=500, y=315)

show_video(label1)

root.mainloop()

cap1.release()
