from PIL import Image, ImageEnhance
from Transforms import RGBTransform
import random
import os
import sys

class Simulator():
    def __init__(self, img_num, png_path):
        self.img_num = img_num
        self.png_path = png_path

        self.img_path = './result/'
        self.check_folder_exists(self.img_path)
        


    def create_data(self):
        png_dirs = os.listdir(self.png_path)

        num_png_dir = len(png_dirs)

        for png_dir in png_dirs:
            self.check_folder_exists(self.img_path + '/' + png_dir)

            for i in range(0, self.img_num):

                png = random.choice(os.listdir(self.png_path + png_dir))
                img = Image.open(os.path.join(self.png_path, png_dir, png))                

                filter_img = self.augment(img)


                filter_img.save('%s/%s/%05d.jpg' % (self.img_path, png_dir, i+50), 'JPEG')
                print('image %s/%s/%05d.jpg saved.' % (self.img_path, png_dir, i+50))
        print ('finish')

    # Check Folder. If the folder is not exist create folder
    def check_folder_exists(self, path):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                print 'create ' + path
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def augment(self, image):
        output_img = self.rotate_image(image)
        output_img = self.brightness(output_img, (0.8, 1.2))
        output_img = self.contrast(output_img, (0.7, 1.3))
        output_img = self.balanceR(output_img, (0.00, 0.03))
        output_img = self.balanceG(output_img, (0.00, 0.03))
        output_img = self.balanceB(output_img, (0.00, 0.03))

        return output_img



    ## rotate Image 
    def rotate_image(self, image):
        img = image
        rotatednum = random.choice([0, 90, 180, 270]) # choice rotation angle randomly
        rotatedImg = img.rotate(rotatednum, expand=True) # roate image
        return rotatedImg



    # r balance
    def balanceR(self, image, filter_num):
        input_im = image
        min_, max_ = filter_num
        num = float("{0:.2f}".format(random.uniform(min_, max_), 1))
        output_im = RGBTransform().mix_with((255, 0, 0), factor=num).applied_to(input_im)

        return output_im

    # g balance
    def balanceG(self, image, filter_num):
        input_im = image
        min_, max_ = filter_num
        num = float("{0:.2f}".format(random.uniform(min_, max_), 1))
        output_im = RGBTransform().mix_with((0, 255, 0), factor=num).applied_to(input_im)

        return output_im


    # b balance
    def balanceB(self, image, filter_num):
        input_im = image
        min_, max_ = filter_num
        num = float("{0:.2f}".format(random.uniform(min_, max_), 1))
        output_im = RGBTransform().mix_with((0, 0, 255), factor=num).applied_to(input_im)

        return output_im



    # brightness filter for object image
    def brightness(self, image, filter_num):
        input_im = image
        min_, max_ = filter_num
        num = round(random.uniform(min_, max_), 1)
        bright = ImageEnhance.Brightness(input_im)
        output_im = bright.enhance(num)
        return output_im

    # contrast filter for object image
    def contrast(self, image, filter_num):
        img = image
        min_, max_ = filter_num
        num = round(random.uniform(min_, max_), 1)
        enhancer = ImageEnhance.Contrast(img)
        cont = enhancer.enhance(num)
        return cont



if __name__ == '__main__':
    final_num = int(sys.argv[1])
    folder_path = sys.argv[2]
    simulator = Simulator(final_num, folder_path)
    simulator.create_data()
