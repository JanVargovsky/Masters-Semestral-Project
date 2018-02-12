import sys
from game import GameApp


def main(argv):
    gameApp = GameApp()
    gameApp.run()
    sys.exit()

if __name__ == "__main__":
    main(sys.argv)