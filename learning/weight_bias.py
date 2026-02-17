import numpy as np
import matplotlib.pyplot as plt

def plot_line(w, b, label, color):
    x = np.linspace(-10, 10, 100)
    # 线性方程: y = w * x + b
    y = w * x + b
    plt.plot(x, y, label=label, color=color, linewidth=2)

plt.figure(figsize=(10, 6))

# 1. 基准线 (w=1, b=0)
plot_line(w=1, b=0, label="Base: w=1, b=0", color='black')

# 2. 改变 Weight (旋转)
# 当 w 变大，线变得更陡峭；当 w 变负，线变方向
plot_line(w=3, b=0, label="Change Weight: w=3 (Steeper)", color='red')
plot_line(w=-0.5, b=0, label="Change Weight: w=-0.5 (Flip)", color='orange')

# 3. 改变 Bias (平移)
# w 不变，只改变 b，线会上下/左右平行移动
plot_line(w=1, b=5, label="Change Bias: b=5 (Shift Up)", color='blue')
plot_line(w=1, b=-5, label="Change Bias: b=-5 (Shift Down)", color='green')

plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.ylim(-10, 10)
plt.grid(True, linestyle='--')
plt.legend()
plt.title("How Weight and Bias affect the Decision Boundary")
plt.show()

