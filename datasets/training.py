from data.preprocessing import array2batch


class TrainData():
    def __init__(self, x, y_tr, y_te):
        
        self.x = x
        self.y_tr = y_tr
        self.y_te = y_te
    
    def get_x_train(self):
        
        return array2batch(self.x)
    
    def get_x_test(self):
        
        return array2batch(self.x)
    
    def get_y_train(self):
        
        return array2batch(self.y_tr)
    
    def get_y_test(self):
       
        return array2batch(self.y_te)
