import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import util
import ipdb
# code from https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
# used to display structural images

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

class Display:
    def __init__(self, filename):
        self.filename = filename
        self.img = util.open_nii_img(filename)

        self.num_dim = len(self.img.shape)
        self.img_idx = 0
        print(self.img.shape)

    def multi_slice_viewer(self):
        remove_keymap_conflicts({'j', 'k', 'h', 'l'})
        fig, ax = plt.subplots()
        if self.num_dim == 3:
            volume = self.img
        else:
            volume = self.img[0]
        ax.volume = volume
        ax.index = 0
        ax.imshow(volume[ax.index])
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        plt.show()

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        elif event.key == 'h':
            self.previous_volume(ax)
        elif event.key == 'l':
            self.next_volume(ax)
        ax.imshow(ax.volume[ax.index])
        fig.canvas.draw()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
        print('new ax.index {}'.format(ax.index))

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

    def previous_volume(self, ax):
        if self.num_dim <= 3:
            return
        self.img_idx = (self.img_idx + 1) % self.img.shape[0]
        ax.volume = self.img[self.img_idx]
        ax.images[0].set_array(self.img[self.img_idx][ax.index])

    def next_volume(self, ax):
        if self.num_dim <= 3:
            return
        self.img_idx = (self.img_idx - 1) % self.img.shape[0]
        ax.volume = self.img[self.img_idx]
        ax.images[0].set_array(self.img[self.img_idx][ax.index])



if __name__ == '__main__':
    d = Display('./sfnwmrda0010100_session_1_rest_1.nii.gz')
    # d = Display('./snwmrda0010001_session_1_rest_2.nii.gz')
    # d = Display('./wssd0010001_session_1_anat.nii.gz')
    # img = util.open_nii_img('./wssd0010001_session_1_anat.nii.gz')
    # multi_slice_viewer(img)
    # plt.show()
    d.multi_slice_viewer()