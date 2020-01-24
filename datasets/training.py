from data.preprocessing import img2batch


class TrainData():
    def __init__(self, x, y_tr, y_te):
        
        self.x = x
        self.y_tr = y_tr
        self.y_te = y_te
    
    def get_x_train(self):
        
        return img2batch(self.x)
    
    def get_x_test(self):
        
        return img2batch(self.x)
    
    def get_y_train(self):
        
        return img2batch(self.y_tr)
    
    def get_y_test(self):
       
        return img2batch(self.y_te)
