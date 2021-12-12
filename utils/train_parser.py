
class Options():
    def __init__(self, trans = '42_AG', method = 'fcn_42_AG_clic', level = 1, 
                 epochs = 200, lr = 1e-2, decay = 1e-4, 
                 dynamic = 0, num_neuron=16):
        self.num_workers = 5
        self.trans = trans
        self.method = method
        self.level = level
        self.dynamic = dynamic
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.num_neuron = num_neuron
        
        self.save_models_path = '/path/to/weights/models_%sn/%s'%(num_neuron, method)
        self.save_reference_path = '/path/to/references/%s'%(method)

        self.approx_path_train = self.save_reference_path + \
        '/level%s_approx_%s_train'%(level, method)
        self.approx_path_test = self.save_reference_path + \
        '/level%s_approx_%s_test'%(level, method)
        
        if self.level == 1:
            self.im_path_train = '/path/to/datasets/clic/centered_ordered_clic_train_dataset'
            self.im_path_test = '/path/to/datasets/clic/centered_ordered_clic_valid_dataset'
            
        else:
            self.im_path_train = self.save_reference_path + \
            '/level%s_approx_%s_train'%(level-1, method)
            self.im_path_test = self.save_reference_path + \
            '/level%s_approx_%s_test'%(level-1, method)

        self.P3_train_path = self.save_reference_path +'/level%s_X3_%s_train'%(level, method)
        self.P3_test_path = self.save_reference_path + '/level%s_X3_%s_test'%(level, method)

        self.P2_train_path = self.save_reference_path + '/level%s_X2_%s_train'%(level, method)
        self.P2_test_path = self.save_reference_path + '/level%s_X2_%s_test'%(level, method)

        self.P1_train_path = self.save_reference_path + '/level%s_X1_%s_train'%(level, method)
        self.P1_test_path = self.save_reference_path + '/level%s_X1_%s_test'%(level, method)

        self.U_train_path = self.save_reference_path + '/level%s_U_%s_train'%(level, method)
        self.U_test_path = self.save_reference_path + '/level%s_U_%s_test'%(level, method)

        self.P3_log_path = self.save_models_path + '/hist_level%s_X3_%s.log'%(level, 
                                                                            method)
        self.P3_model_path = self.save_models_path + '/level%s_X3_%s.h5'%(level, 
                                                                          method)

        self.P2_log_path = self.save_models_path + '/hist_level%s_X2_%s.log'%(level, 
                                                                            method)
        
        self.P2_model_path = self.save_models_path + '/level%s_X2_%s.h5'%(level, 
                                                                              method)

        self.P1_log_path = self.save_models_path + '/hist_level%s_X1_%s.log'%(level, 
                                                                            method)
        
        self.P1_model_path = self.save_models_path + '/level%s_X1_%s.h5'%(level, 
                                                                              method)

        self.U_log_path = self.save_models_path + '/hist_level%s_U_%s.log'%(level, 
                                                                          method)

        self.U_model_path = self.save_models_path + '/level%s_U_%s.h5'%(level, 
                                                                            method)

    def configure(self, params):
        for k in params.keys():
            self.__dict__[k] = params[k]