from games import Game, TrainingGame
import time

def main():
    start = time.time()
    training_game = TrainingGame()
    training_game.training_session(300000000)
    end = time.time()   
    print(f"Total training seesion time: {end-start}")


if __name__ == '__main__':
    main()