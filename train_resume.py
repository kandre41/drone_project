from ultralytics import YOLO
def main():
# Load a model
    model = YOLO("runs\\pose\\train12\\weights\\last.pt")  # load a partially trained model

# Resume training
    results = model.train(resume=True)
if __name__ == '__main__':
    main()