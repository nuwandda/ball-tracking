import PySimpleGUI as sg
from process import Process
from PIL import Image
import io
import cv2
import imutils


class Interface:
    def __init__(self):
        self.initUI()

    def initUI(self):
        layout = [[sg.Text('Improvise. Adapt. Overcome.', size=(30, 1), font=("Helvetica", 25))],
                  [sg.Image(filename='', key='frame'), sg.Image(filename='', key='dst_img')],
                  [sg.Text('Choose A Folder', size=(35, 1))],
                  [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
                   sg.InputText('Default Folder', key='path'), sg.FileBrowse()],
                  [sg.Button('Load'), sg.Button('Start'), sg.Button('Save'), sg.Button('About'), sg.Button('Quit')]]

        window = sg.Window('My Basketball Coach', default_element_size=(80, 1)).Layout(layout)
        window.Move(0, 0)
        process = None
        about_window_active = False

        while True:
            event, values = window.ReadNonBlocking()

            if event == 'Load':
                process = Process(values['path'])
            elif event == 'Start':
                sg.Popup(
                    'Please select with the order below:\nTop Left, Bottom Left, Bottom Right and Top Right.\nHit "e" for selection, "r" for select again.')
                process.getInitFourPoints
                sg.Popup('Please select the backboard.\nHit "Enter" for selection, "c" for select again.')
                process.getInitROI()
                process.start()
            elif event == 'Quit':
                break
            elif event == 'Save':
                save_path = sg.PopupGetFile('Choose', save_as=True)
                cv2.imwrite(save_path + ".png", process.dst_image_clone)
            elif event == 'About' and not about_window_active:
                about_window_active = True
                window.Hide()
                layout_about = [[sg.Text('About')], [sg.Button('Exit')]]
                about_window = sg.Window('About', layout_about)
                while True:  
                    event2, values2 = about_window.Read()  
                    if event2 == 'Exit':  
                        about_window.Close()  
                        about_window_active = False  
                        window.UnHide()  
                        break  

            if process is not None:
                dst_image = cv2.cvtColor(process.dst_image_clone, cv2.COLOR_BGR2RGB)
                # dst_image = imutils.resize(dst_image, width=400)
                img = Image.fromarray(dst_image)  # create PIL image from frame
                bio = io.BytesIO()  # a binary memory resident stream
                img.save(bio, format='PNG')  # save image as png to it
                imgbytes = bio.getvalue()  # this can be used by OpenCV hopefully
                window.FindElement('dst_img').Update(data=imgbytes)
