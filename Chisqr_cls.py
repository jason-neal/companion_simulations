

class ChisqrAnalyser(object):

    def __init__(self, star, num, chip):
        self.star = star
        self.num = num
        self.chip = chip

    # Get parameters from model



    def bhm_chisqr(self):
        print(self.star, self.num, self.chip)
        print("model = bhm")

    def tcm_chisqr(self):
        print(self.star, self.num, self.chip)
        print("model = tcm")


if __name__ == "__main__":
    c = ChisqrAnalyser("HD211847", "1", 1)
    chisqr = c.bhm_chisqr
    chisqr2 = c.tcm_chisqr
