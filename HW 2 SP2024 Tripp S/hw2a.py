import numpy as np

"""The Probability function sets up mu and sigma as a tuple with args and then creates an if statement that declares
if GT is greater than c or less than c, a or b will be given different values. as well as sets up n for the simpsons
rule. As well as the simpsons rule itself"""


def Probability(PDF, args, c, GT=True):
    mu, sigma = args
    if GT:
        a = mu - 5 * sigma
        b = c
    else:
        a = c
        b = mu + 5 * sigma

    n = 10_000
    h = (b - a) / n

    x = np.linspace(a, b, n + 1)
    y = PDF((x, mu, sigma))

    integral = (h / 3) * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])

    return integral


"""The normal pdf function just calls back to the probability density function given in the problem. """


def normal_pdf(data):
    x, mu, sigma = data
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


"""The main function just provides the values for mu, sigma, and c to then use in the probability functions set
earlier in the program."""


def main():

    mu1, sigma1 = 100, 12.5
    c1 = 105
    prob1 = Probability(normal_pdf, (mu1, sigma1), c1, GT=False)

    mu2, sigma2 = 100, 3
    c2 = mu2 + 2 * sigma2
    prob2 = Probability(normal_pdf, (mu2, sigma2), c2, GT=True)

    print(f'P(x<{c1:.2f}|N({mu1},{sigma1}))={prob1:.2f}')
    print(f'P(x>{c2:.2f}|N({mu2},{sigma2}))={prob2:.2f}')


if __name__ == "__main__":
    main()
