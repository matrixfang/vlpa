import matplotlib.pyplot as plt
import pickle
import numpy as np


def pos_compare_plot():
    with open('pos_compare.dat', 'r') as f:
        x = pickle.load(f)
        nmi_a = pickle.load(f)
        nmi_b = pickle.load(f)
    print(x, nmi_a, nmi_b)

    plt.figure(1)
    plt.plot(x, nmi_a, label='posshrink')
    plt.plot(x, nmi_b, label='noposshrink')
    plt.legend(loc='upper left')
    plt.savefig('pos_compare.png')


def convergence_plot():
    with open('convergence_test.dat', 'r') as f:
        log_k = pickle.load(f)
        log_values = pickle.load(f)
    x = np.array(log_k[-20:])
    y = np.array(log_values[-20:])
    print(x)
    print(y)
    fit = np.polyfit(x, y, 1)
    fit_function = np.poly1d(fit)
    k_str = "%.4f" % fit[0]
    print(k_str)
    xant = log_k[12]
    yant = fit_function(log_k[12])
    plt.figure(1)
    plt.plot(log_k[5:35], fit_function(log_k[5:35]), '--', label='fitting line')
    plt.plot(log_k, log_values, label='log differences')
    plt.annotate(r'$slope =$' + k_str, xy=(xant, yant), xytext=(xant + 0.2, yant + 0.2),
                 fontsize=14, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel(r'$\log( k)$')
    plt.ylabel(r'$\log(|Q_k-Q_{opt}|)$')
    plt.legend()
    plt.savefig('convergence_plot.png')


def method_adjust_plot():
    with open('method_adjust.dat', 'r') as f:
        log_a_values = pickle.load(f)
        log_b_values = pickle.load(f)
        log_c_values = pickle.load(f)
        log_d_values = pickle.load(f)
        log_e_values = pickle.load(f)
        log_f_values = pickle.load(f)
    plt.figure(1)
    log_a = [np.log10(v + 1) for v in range(len(log_a_values))]
    log_b = [np.log10(v + 1) for v in range(len(log_b_values))]
    log_c = [np.log10(v + 1) for v in range(len(log_c_values))]
    log_d = [np.log10(v + 1) for v in range(len(log_d_values))]
    log_e = [np.log10(v + 1) for v in range(len(log_e_values))]
    log_f = [np.log10(v + 1) for v in range(len(log_f_values))]
    plt.plot(log_a, log_a_values, label='both 0.5')
    plt.plot(log_b, log_b_values, label='both 0.9')
    plt.plot(log_c, log_c_values, label='nothing 0.5')
    plt.plot(log_d, log_d_values, label='nothing 0.9')
    plt.plot(log_e, log_e_values, label='normalize 0.5')
    plt.plot(log_f, log_f_values, label='normalize 0.9')
    plt.legend()
    plt.savefig('update strategy.png')


def plot():
    x = [1,2,3,4,5]
    y = [3,5,6,7,9]
    z = [6,7,10,12,13]
    plt.plot(x,y,label='line a')
    plt.plot(x,z, label='line b')
    plt.style
    plt.savefig('example.eps')

plot()