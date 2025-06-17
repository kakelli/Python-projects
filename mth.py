import math 
import tkinter as tk

def click(event):
    text = event.widget.cget("text")
    if text == '=':
        try:
            result = eval(screen.get())
            screen_var.set(result)
        except Exception as e:
            screen_var.set("Error")
    elif text == 'C':
        screen_var.set('')
    else:
        screen_var.set(screen_var.get()+text)

root = tk.Tk()
root.title("Calculator")

screen_var = tk.StringVar()
screen = tk.Entry(root, textvar = screen_var, font = "Arial 20", justify = 'right')
screen.pack(fill = 'both', ipadx = 8, pady = 10, padx = 10)

button_frame = tk.Frame(root)
button_frame.pack()

buttons = [
    ['7','8','9','/'],
    ['4','5','6','*'],
    ['1','2','3','+'],
    ['0','.','=','-'],
    ['C','sqrt','^','%']
]

for row in buttons:
    frame = tk.Frame(button_frame)
    frame.pack(side = 'top')
    for btn_text in row:
        def make_command(x =btn_text):
            if x == 'sqrt':
                return  lambda e = None:screen_var.set(math.sqrt(float(screen.get())))
            elif x == '^':
                return lambda e =None:screen_var.set(screen.get()+"**")
            else:
                return click
        b = tk.Button(frame, text = btn_text, font = "Arial 18", width = 5, height = 2)
        b.pack(side= 'left', padx =5, pady = 5)
        b.bind("<Button-1>", make_command())

root.mainloop()
