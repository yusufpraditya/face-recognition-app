class Sembarang():
    test_list = []
    _test_val = 123

    def set_list(self):
        self.test_list.append(self)
        self.test_list[0]._test_val = 1

        self.test_list.append(self)
        self.test_list[1]._test_val = 2

        self.test_list.append(self)
        self.test_list[2]._test_val = 3

        print(self.test_list[0]._test_val)
        print(self.test_list[2]._test_val)
        
    
    def update_val(self):
        self.test_val += 123

    def get_list(self):
        return self.test_list

var_a = Sembarang()
var_a.set_list()

var_b = var_a.get_list()[0]
var_c = var_a.get_list()[1]
var_d = var_a.get_list()[2]
#del var_d.test_list[0]
print(var_a._test_val)


some_list = [1,2,3,4]
for i in range(len(some_list) + 1):
    print(some_list[i])