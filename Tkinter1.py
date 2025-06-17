import tkinter as tk
#Main window
w = tk.Tk()
w.title("Caesar")
w.geometry("300x200")
#Add label 
l = tk.Label(w, text = "This is black mukku")
l.pack()
#   Button
def hello():
    l.config(text="TOuched this balck mukku")
b = tk.Button(w,text = "CLick this mukku", command=hello)
b.pack()
w.mainloop()
