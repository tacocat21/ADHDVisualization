from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import util
import numpy as np
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
    def __init__(self, input_val, mask=None):
        # input_val can be filename or image
        if type(input_val) is str:
            self.filename = input_val
            self.img = util.open_nii_img(input_val)
        else:
            self.filename = ''
            self.img = input_val

        self.num_dim = len(self.img.shape)
        self.img_idx = 0
        print(self.img.shape)
        self.mask = mask
        self.alpha = 0.6

    def multi_slice_viewer(self):
        remove_keymap_conflicts({'j', 'k', 'h', 'l'})
        fig, ax = plt.subplots()
        if self.num_dim == 3:
            volume = self.img
        else:
            volume = self.img[0]
        ax.volume = volume
        ax.index = 0
        self.im = ax.imshow(volume[ax.index])
        cbar = fig.colorbar(self.im, extend='both', spacing='uniform',
                            shrink=0.9, ax=ax)
        # if self.mask is not None:
        #     ax.imshow(self.mask[ax.index], alpha=0.5, cmap='jet')
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
        if self.mask is not None:
            ax.images[0].set_array(volume[ax.index] *(1.0 - self.alpha) + self.alpha * 255 *self.mask[ax.index])
        else:
            ax.images[0].set_array(volume[ax.index])
        print('new ax.index {}'.format(ax.index))

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        # ax.images[0].set_array(volume[ax.index])
        if self.mask is not None:
            print('using mask')
            ax.images[0].set_array(volume[ax.index] *(1.0 - self.alpha) + self.alpha * 255 *self.mask[ax.index])
        else:
            ax.images[0].set_array(volume[ax.index])

        print('new ax.index {}'.format(ax.index))

    def previous_volume(self, ax):
        if self.num_dim <= 3:
            return
        self.img_idx = (self.img_idx + 1) % self.img.shape[0]
        ax.volume = self.img[self.img_idx]
        ax.images[0].set_array(self.img[self.img_idx][ax.index])
        print('new img_index {}'.format(self.img_idx))


    def next_volume(self, ax):
        if self.num_dim <= 3:
            return
        self.img_idx = (self.img_idx - 1) % self.img.shape[0]
        ax.volume = self.img[self.img_idx]
        ax.images[0].set_array(self.img[self.img_idx][ax.index])
        print('new img_index {}'.format(self.img_idx))


def save_3d_image_display(img, title, filename):
    w = 15
    h = 15
    fig = plt.figure(figsize=(10, 10))
    columns = 9
    rows = 10
    plt.title(title)
    plt.axis('off')
    cdict3 = {'red': ((0.0, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.5, 0.8, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 0.4, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.5, 0.8, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 0.4, 1.0)),
              'blue': ((0.0, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.5, 0.8, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 0.4, 1.0))
              }
    # blue_red1 = LinearSegmentedColormap('BlueRed1', cdict3)
    cmap = plt.get_cmap('viridis')
    for i in range(1, columns * rows + 1):
        if i == img.shape[0]:
            break
        _img = img[i-1]
        fig.add_subplot(rows, columns, i)
        im = plt.imshow(_img, cmap=cmap)
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        # fig.colorbar(im, extend='both', spacing='uniform',
        #          shrink=0.9)
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(im, shrink=0.9, cax=cb_ax)
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    # d = Display('./sfnwmrda0010100_session_1_rest_1.nii.gz')
    # d = Display('./snwmrda0010001_session_1_rest_2.nii.gz')
    # d = Display('./wssd0010001_session_1_anat.nii.gz')
    # img = util.open_nii_img('./wssd0010001_session_1_anat.nii.gz')
    # multi_slice_viewer(img)
    # plt.show()
    # d.multi_slice_viewer()

    cam = np.load('cam.npy')
    print(cam.shape)
    save_3d_image_display(cam, 'cam.png')