class Sembarang():
    test_list = []
    test_val = 123

    def set_list(self):
        self.test_val += 1
        self.test_list.append(self)
        self.test_val += 1
        self.test_list.append(self)
        self.test_val += 1
        self.test_list.append(self)
    
    def update_val(self):
        self.test_val += 123

    def get_list(self):
        return self.test_list

var_a = Sembarang()
var_a.set_list()

var_b = var_a.get_list()[0]
var_c = var_a.get_list()[1]
var_d = var_a.get_list()[2]
del var_d.test_list[0]
print(len(var_d.test_list))
