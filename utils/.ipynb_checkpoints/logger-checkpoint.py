class Log(object):
    def __init__(self, log_file_path):
        self.fout = open(log_file_path, 'w')

    def log_training(self, loss, epoch):
        self.fout.write('\n\n### Training Epoch: ' + str(epoch) + '\n')
        self.fout.write(str(loss) + '\n')


    def log_validation(self, loss, epoch):
        self.fout.write('\n\n### Validation Epoch: ' + str(epoch) + '\n')
        self.fout.write(str(loss) + '\n')

    def close(self):
        self.fout.close()
