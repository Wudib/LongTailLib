# LongTailLib
联邦长尾学习评测平台

数据集生成部分

1、	global（大多数论文采用的方法）
先长尾分布（IF因子）再进行Direchlet分布

参数设置：

1、iid：独立同分布，noniid：非独立同分布

2、"-"：客户端数据量不平衡，balance：客户端数据量平衡

3、dir：Dirichlet 分布划分、pat：病理非 IID、exdir：扩展 Dirichlet 策略，进一步增强非 IID 特性

4、longtail：使用长尾分布

5、global：全局长尾分布、local：本地长尾分布

6、不平衡因子IF

7、Dirichlet中的alpha设置

8、客户端数量

调用语句：

python dataset/generate_Cifar10.py noniid – dir longtail global 50 0.5 20  

<img width="865" height="356" alt="image" src="https://github.com/user-attachments/assets/cc2fa17e-6d6f-4ae2-8fe1-1760e1eea3e7" />

