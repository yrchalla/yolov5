import matplotlib.pyplot as plt

train_loss = []
test_loss = []
top1_acc = []
epochs = []

with open('data.txt', 'r') as file:
    next(file)  # Skip the header line
    i = 1
    for line in file:
        values = line.strip().split()
        train_loss.append(float(values[2]))
        test_loss.append(float(values[3]))
        top1_acc.append(float(values[4]))
        epochs.append(i)
        i += 1

# Plotting the data
plt.figure(figsize=(10, 5))

# Train loss plot
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Test loss plot
plt.subplot(1, 3, 2)
plt.plot(epochs, test_loss)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Top 1 accuracy plot
plt.subplot(1, 3, 3)
plt.plot(epochs, top1_acc)
plt.title('Top 1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Adjusting the layout and displaying the plots
plt.tight_layout()
plt.show()
