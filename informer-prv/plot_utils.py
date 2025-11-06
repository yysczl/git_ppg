import matplotlib.pyplot as plt

# 绘制训练过程
def plot_training_process(train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses):
    plt.figure(figsize=(15, 5))
    
    # 绘制损失
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
    # 绘制MAE
    plt.subplot(1, 3, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE vs. Epoch')
    plt.legend()
    
    # 绘制RMSE
    plt.subplot(1, 3, 3)
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(val_rmses, label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.show()

# 绘制预测结果
def plot_predictions(predictions, targets):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. True Values')
    plt.savefig('predictions.png')
    plt.show()