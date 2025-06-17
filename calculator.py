import tkinter 

# Create the main window
root = tkinter.Tk()
root.title("My First Tkinter App")
root.geometry("300x200")  # width x height

# Add a label
label = tkinter.Label(root, text="Hello, Tkinter!")
label.pack()

# Add a button
def say_hello():
    label.config(text="Button Clicked!")

button = tkinter.Button(root, text="Click Me", command=say_hello)
button.pack()

# Start the event loop
root.mainloop()
