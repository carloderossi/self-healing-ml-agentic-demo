from model.train_model import train

class Retrainer:
    def retrain(self, train_path="data/train.csv"):
        print("[RETRAIN] Starting retraining...")
        new_acc = train(train_path)
        print(f"[RETRAIN] New validation accuracy: {new_acc}")
        return new_acc