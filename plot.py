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


def fit_plot(x,y,plots='False',upmove=0):
    print(x)
    print(y)
    if plots=='False':
        def plots():
            pass
    elif plots=='True':
        def plots():
            plt.plot(x, y, 's')
    else:
        raise('plots Must be Boolean')

    fit = np.polyfit(x, y, 1)
    fit[1] = fit[1] + upmove
    print(fit)
    fit_function = np.poly1d(fit)
    k_str = "%.4f" % fit[0]
    print(k_str)
    num=len(x)
    xant = x[num/2]
    yant = fit_function(x[num/2])

    plt.figure(1)
    plots()
    plt.plot(x, fit_function(x), '--')
    plt.annotate(r'$slope =$' + k_str, xy=(xant, yant), xytext=(xant + 0.1, yant + 0.1),
                 fontsize=12, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    return fit[0]


def fit_plot_all():
    with open('convergence_test.dat', 'r') as f:
        log_k = pickle.load(f)
        log_values = pickle.load(f)
    x = np.array(log_k[-20:])
    y = np.array(log_values[-20:])
    fit_plot(x,y)
    plt.show()
    pass
