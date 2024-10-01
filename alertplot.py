import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tkinter as tk
from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg )
from matplotlib.figure import Figure

class AlertPlot(tk.Frame):
    def __init__(self, root, w = 15, h = 5, hideaxes = False):
        tk.Frame.__init__(self, root)
        self.w = w
        self.h = h
        self.fig = Figure(figsize=(3, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.currentGrid = []
        self.hideaxes = hideaxes

    def display(self, gridValues : list):
        self.currentGrid = gridValues
        if hasattr(self, 'ax'):
            self.ax.cla()
        else:
            self.ax = self.fig.add_subplot()
        #show or hide axes plot
        self.hideAxis(self.hideaxes)
        self.ax.set_xlim([0, self.w])
        self.ax.set_xticks([])
        self.ax.set_ylim([0+0.5, self.h+0.5])
        self.ax.set_ylabel('teeth')
        x = 0
        y = 0
        for col in gridValues:
            for sample in col:
                self.ax.add_patch(Rectangle((x, y+0.5), 1, 1, color=sample))
                y += 1
            x += 1
            y = 0
        self.canvas.draw_idle()

    def appendColumn(self, column : list):
        self.currentGrid.append(column)
        if len(self.currentGrid) > self.w:
            index = self.currentGrid[0]
            self.currentGrid.remove(index)
        self.display(self.currentGrid)

    def clearGridData(self):
        self.currentGrid = []

    def hideAxis(self, s : bool):
        if s:
            self.ax.set_axis_off()
        else:
            self.ax.set_axis_on()
        self.canvas.draw()

    def getGrid(self) -> list:
        return self.currentGrid

alert = None
grid = []

def clicked():
    elem = [[1, 1, 1], [0.7, 0.1, 0]]
    alert.appendColumn(elem)

if __name__ == "__main__":
    root = tk.Tk()
    root.title('prova')

    grid.append([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0.3, 0.2, .5]])
    grid.append([[1, 1, 1], [1, 1, 0]])

    alert = AlertPlot(root, 10, 4)
    alert.pack(side=tk.TOP)
    button = tk.Button(root, text = "premi", command=clicked)
    button.pack(side=tk.TOP)
    alert.display(grid)

    root.mainloop()