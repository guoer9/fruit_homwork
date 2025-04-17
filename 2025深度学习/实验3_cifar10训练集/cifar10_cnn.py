import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# 在这里设置matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置

import matplotlib.pyplot as plt  # 注意这行必须在设置后端之后
import numpy as np
import time
from torch.utils.data import DataLoader

# 从models.py导入模型和模型管理器
from models import ModelManager, get_first_conv_layer

# 1. 完成CIFAR-10数据集的数据读取，并完成数据集的初步探索

# 定义数据转换
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # 随机裁剪
    transforms.RandomHorizontalFlip(),       # 随机水平翻转
    transforms.ToTensor(),                   # 转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


batch_size = 128


# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size , shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size , shuffle=False, num_workers=2)

# CIFAR-10的类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 数据集探索：显示一些训练图像
def imshow(img):
    img = img / 2 + 0.5  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# 获取一些随机的训练图像
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 在代码开头添加一个标志
SKIP_VISUALIZATION = True

# 然后在可视化代码部分添加条件判断
if not SKIP_VISUALIZATION:
    plt.figure(figsize=(10, 4))
    imshow(torchvision.utils.make_grid(images[:8]))
    plt.show()

# 打印标签
print(' '.join(f'{classes[labels[j]]}' for j in range(8)))

# 实例化模型管理器
model_manager = ModelManager()

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")




# 2. 训练函数
def train_model(model, trainloader, testloader, criterion, optimizer, epoch):
    # 记录训练与测试的损失和准确率
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    # 记录训练时间
    start_time = time.time()
    
    for epoch in range(epoch):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计训练损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 每100批次打印一次信息
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}, Acc: {100. * correct / total:.3f}%')
                running_loss = 0.0
        
        # 计算整个训练集的准确率
        train_acc = 100. * correct / total
        train_accs.append(train_acc)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # 不计算梯度
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        test_accs.append(test_acc)
        
        # 记录平均损失
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))
        
        print(f'Epoch {epoch + 1}: Train Acc = {train_acc:.3f}%, Test Acc = {test_acc:.3f}%')
    
    end_time = time.time()
    print(f'训练完成！总耗时: {end_time - start_time:.2f} 秒')
    
    return train_losses, test_losses, train_accs, test_accs

# 3. 训练不同模型的函数
def train_and_evaluate_model(model_name, epoch, lr=0.001):
    print(f"\n{'='*20} 训练模型: {model_name} {'='*20}")
    
    # 获取模型
    model = model_manager.get_model(model_name).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练模型
    train_losses, test_losses, train_accs, test_accs = train_model(
        model, trainloader, testloader, criterion, optimizer, epoch
    )
    
    # 评估模型
    evaluate_model(model, testloader)
    
    # 可视化训练过程
    visualize_training_process(train_losses, test_losses, train_accs, test_accs, epoch, model_name)
    
    # 保存模型
    save_path = f'cifar10_{model_name}_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存至 {save_path}')
    
    return model, train_accs[-1], test_accs[-1]

# 4. 可视化训练过程
def visualize_training_process(train_losses, test_losses, train_accs, test_accs, num_epochs, model_name):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, num_epochs + 1), test_losses, 'r-', label='测试损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.title(f'{model_name} - 训练和测试损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, 'b-', label='训练准确率')
    plt.plot(range(1, num_epochs + 1), test_accs, 'r-', label='测试准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.title(f'{model_name} - 训练和测试准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # 替换show为savefig
    plt.savefig(f'{model_name}_training_process.png')
    plt.close()  # 关闭图形，释放内存

# 5. 模型评估
def evaluate_model(model, testloader):
    # 设置为评估模式
    model.eval()
    
    # 初始化混淆矩阵
    confusion_matrix = torch.zeros(10, 10)
    
    # 不计算梯度
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 更新混淆矩阵
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # 计算每个类别的准确率
    class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    plt.xticks(np.arange(10), classes, rotation=45)
    plt.yticks(np.arange(10), classes)
    
    # 添加数字标签
    thresh = confusion_matrix.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f'{int(confusion_matrix[i, j])}',
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    # 显示每个类别的准确率
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_accuracy.cpu().numpy())
    plt.title('每个类别的准确率')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.show()
    
    # 输出总体准确率
    print(f'总体测试准确率: {100 * confusion_matrix.diag().sum() / confusion_matrix.sum():.2f}%')
    
    # 输出每个类别的准确率
    for i, class_name in enumerate(classes):
        print(f'{class_name} 类别准确率: {100 * class_accuracy[i]:.2f}%')

# 6. 可视化第一层卷积核
def visualize_filters(model, model_name):
    # 获取第一个卷积层的权重
    filters = get_first_conv_layer(model, model_name).cpu().numpy()
    
    # 对过滤器进行归一化以便可视化
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    
    # 显示前16个过滤器
    n_filters = min(16, filters.shape[0])
    plt.figure(figsize=(10, 8))
    
    for i in range(n_filters):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.transpose(filters[i], (1, 2, 0)))
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    
    plt.tight_layout()
    plt.suptitle(f'{model_name} - 第一层卷积核可视化', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

# 7. 比较不同模型的性能
def compare_models(models_to_train, epoch, lr=0.001):
    results = {}
    
    for model_name in models_to_train:
        _, train_acc, test_acc = train_and_evaluate_model(model_name, epoch, lr)
        results[model_name] = {
            'train_acc': train_acc,
            'test_acc': test_acc
        }
    
    # 可视化比较结果
    plt.figure(figsize=(12, 6))
    
    model_names = list(results.keys())
    train_accs = [results[model]['train_acc'] for model in model_names]
    test_accs = [results[model]['test_acc'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='训练准确率')
    plt.bar(x + width/2, test_accs, width, label='测试准确率')
    
    plt.xlabel('模型')
    plt.ylabel('准确率 (%)')
    plt.title('不同模型的训练和测试准确率比较')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results

# 8. 加载已训练模型并进行推理
def load_model_and_predict(model_name, images):
    # 获取模型结构
    model = model_manager.get_model(model_name)
    
    # 加载已训练的权重
    model_path = f'cifar10_{model_name}_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # 进行推理
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    return predicted

# 示例用法
if __name__ == "__main__":
    # 列出所有可用的模型
    print("可用模型:", model_manager.list_available_models())
    
    # 选择要训练的模型
    # models_to_train = ['custom_cnn', 'lenet', 'alexnet', 'resnet18', 'vgg16']
    models_to_train = ['custom_cnn', 'lenet', 'alexnet', 'resnet18', 'vgg16']  
    
    # 训练并比较选定的模型
    epoch = 20
    results = compare_models(models_to_train, epoch)
    
    # 打印比较结果
    print("\n模型性能比较:")
    for model_name, metrics in results.items():
        print(f"{model_name}: 训练准确率 = {metrics['train_acc']:.2f}%, 测试准确率 = {metrics['test_acc']:.2f}%") 