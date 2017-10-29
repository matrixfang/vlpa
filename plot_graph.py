import matplotlib.pyplot as plt
import pickle
def pos_compare_plot():
    with open('pos_compare.dat','r') as f:
        x = pickle.load(f)
        nmi_a = pickle.load(f)
        nmi_b = pickle.load(f)
    print(x, nmi_a, nmi_b)

    plt.figure(1)
    plt.plot(x, nmi_a, label='posshrink')
    plt.plot(x, nmi_b, label='noposshrink')
    plt.legend(loc='upper left')
    plt.savefig('pos_compare.png')

pos_compare_plot()