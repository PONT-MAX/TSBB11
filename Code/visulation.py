import wx

class ImageFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self ,None ,title = "Picture Viewer")
        self.menubar = wx.MenuBar()
        self.file = wx.Menu()
        self.SetMenuBar(self.menubar)
        self.menubar.Append(self.file ,'&File')
        self.get = self.file.Append(wx.ID_ANY ,'Get &Picture')
        self.Bind(wx.EVT_MENU ,self.getpicture ,self.get)
        self.picturepanel = wx.Panel(self ,style = wx.SUNKEN_BORDER)
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.getpicbut = wx.Button(self ,wx.ID_ANY ,label = 'Get Picture')
        self.getpicbut.Bind(wx.EVT_LEFT_DOWN ,self.getpicture)
        self.vbox.Add(self.getpicbut ,proportion=0 ,flag = wx.EXPAND)
        self.vbox.Add(self.picturepanel ,proportion=1 ,flag = wx.EXPAND)
        self.SetSizer(self.vbox)
        self.Show()
        self.img =0
        #wx.App(False)  # the wx.App object must be created first.
        print(wx.GetDisplaySize())
        self.display_size_x = wx.GetDisplaySize()[0]
        self.display_size_y = wx.GetDisplaySize()[1]



    def getpicture(self, event):
        filedialog = wx.FileDialog(self,
                                    message = 'Choose an image to open',
                                    defaultDir = '',
                                    defaultFile = '',
                                    wildcard = 'Supported Image files (*.gif;*.png;*.jpg;*.bmp;*.jpeg) | *.gif; *.png; *.jpg;;*.jpeg *.bmp',
                                    style = wx.OPEN)
        if filedialog.ShowModal() == wx.ID_OK:
            if self.img:
                self.img.Destroy()
            self.path = filedialog.GetPath()
            self.bmp = wx.Bitmap(self.path)
            self.img = wx.StaticBitmap(self.picturepanel, wx.ID_ANY, self.bmp)
            self.picturepanel.Refresh()
            imgsize = self.img.Size
            x = imgsize[0] + 12
            y = imgsize[1] + 80
            print(self.Size.GetSize())
            self.Size = (self.display_size_x,self.display_size_y)


app = wx.App(redirect=False)
frame = ImageFrame()
app.MainLoop()