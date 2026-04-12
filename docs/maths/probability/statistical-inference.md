# 统计推断

在[上一章](probability-basics.md)里，我们学习了概率分布的描述方法，给定分布参数，计算出相应的概率。但在实际应用中，人们还面临着另一类相反的问题：给定观测数据，如何推断分布的参数？这就是**统计推断**（Statistical Inference）的要解决的问题。

统计推断是机器学习中模型训练的理论基础。当我们训练一个模型时，本质上就是在做统计推断：从有限的训练数据推断模型参数，然后用这些参数预测新数据。本章将介绍两类主要的推断方法：**点估计**和**区间估计**，以及两种统计哲学：**频率学派**和**贝叶斯学派**。

## 点估计

**估计**（Estimation）是指用样本数据推断总体参数的值或范围。简单来说，我们无法直接观测整个总体（比如全部人口），只能从有限的样本中推断总体特征，这个过程就是估计。

**点估计**（Point Estimation）是用样本数据计算出一个具体数值，作为总体参数的估计值。譬如，用样本均值 $\bar{x}$ 估计总体均值 $\mu$、用样本比例 $\hat{p}$ 估计总体比例 $p$、用样本方差 $s^2$ 估计总体方差 $\sigma^2$，等等，这些都属于点估计的范畴。

点估计要解决的核心问题是**给定一个参数，应该用什么方法从样本数据计算出估计值**。同一个参数往往有多种可能的估计方式，譬如衡量一个国家的居民收入，估计总体均值，既可以用样本均值，也可以用样本中位数，甚至可以用样本的最大值和最小值的平均。不同的估计方法会产生不同的结果，有的估计更准确，有的更稳定，有的计算更简单。因此，我们需要一套原则来评判估计量的好坏，并据此选择或构造最优估计量的方法。统计学中最常用的两种构造估计量的方法是**最大似然估计**和**最大后验估计**。

### 最大似然估计（MLE）

**最大似然估计**（Maximum Likelihood Estimation, MLE）是最经典的点估计方法，其核心思想是**在所有可能的参数值中，选择那个让"当前观测数据最可能出现"的参数值作为估计结果**。

举个直观的例子：假设你抛了 10 次硬币，观察到 8 次正面、2 次反面。现在要估计这枚硬币正面朝上的概率 $p$。$p$ 可能是 0.5 、0.6 、0.7 、0.8 等任何值，但哪个值最能解释"8 正 2 反"这个观测结果？直觉告诉我们，$p=0.8$ 最合理，因为如果正面概率真的是 0.8，那么出现"8 正 2 反"的概率确实最大。这就是最大似然估计的思想：**让数据"自己说话"，选择最能解释当前数据的参数**。数学上，给定观测数据 $X = \{x_1, x_2, \ldots, x_n\}$，MLE 寻找参数 $\theta$ 使得：

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta)$$

其中 $L(\theta) = P(X|\theta)$ 称为**似然函数**，它表示"在参数 $\theta$ 下，观测到数据 $X$ 的概率"。注意似然函数与概率函数的区别：概率是"给定参数，求数据出现的可能性"，而似然是"给定数据，衡量参数的合理性"。最大似然估计的计算遵循一套标准的数学流程：

**第一步：写出似然函数。** 假设样本是独立同分布（即每个样本互不影响、且都来自同一个概率分布）的，每个观测 $x_i$ 出现的概率是 $P(x_i|\theta)$，那么所有观测同时出现的概率就是各概率的乘积：

$$L(\theta) = \prod_{i=1}^n P(x_i|\theta)$$

**第二步：取对数得到对数似然函数。** 直接处理乘积形式的似然函数很麻烦（多个小数相乘会变得极小），而且求导时乘积法则也很复杂。取对数可以把乘积变成加法，既方便计算又不影响最大值的位置（因为对数是单调递增函数）：

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log P(x_i|\theta)$$

**第三步：对参数求导，令导数为零。** 要找函数的最大值，标准方法是求导并令导数为零，解出使函数达到极值的参数值：

$$\frac{\partial \ell(\theta)}{\partial \theta} = 0$$

**第四步：解方程得到估计值。** 从第三步的方程中解出 $\theta$，即为最大似然估计 $\hat{\theta}_{MLE}$。

下面用两个具体的案例来说明 MLE 的工作原理：

- **案例 1**（伯努利分布的 MLE）：伯努利分布是最简单的概率分布 —— 只有两种可能结果（如抛硬币的正反面），参数 $p$ 表示其中一种结果出现的概率。以下代码将模拟抛硬币实验，演示如何用 MLE 从观测数据估计 $p$，并通过可视化似然函数曲线和数学推导来验证结果。

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    # MLE 估计：伯努利分布
    def bernoulli_mle(data):
        """伯努利分布参数 p 的 MLE 估计"""
        return np.mean(data)  # 正面的比例

    # 模拟抛硬币
    true_p = 0.8
    n = 100
    flips = np.random.binomial(1, true_p, n)

    # MLE 估计
    p_mle = bernoulli_mle(flips)

    print(f"真实参数： p = {true_p}")
    print(f"观测数据： {n} 次抛掷，{flips.sum()} 次正面")
    print(f"MLE 估计： p̂ = {p_mle:.4f}")
    print()

    # 可视化似然函数
    p_values = np.linspace(0.01, 0.99, 100)
    k = flips.sum()

    # 似然函数： L(p) = p^k * (1-p)^(n-k)
    # 对数似然： l(p) = k*log(p) + (n-k)*log(1-p)
    log_likelihood = k * np.log(p_values) + (n - k) * np.log(1 - p_values)

    plt.figure(figsize=(10, 5))
    plt.plot(p_values, log_likelihood, 'b-', linewidth=2)
    plt.axvline(p_mle, color='r', linestyle='--', label=f'MLE: p = {p_mle:.2f}')
    plt.axvline(true_p, color='g', linestyle=':', label=f'真实值： p = {true_p}')
    plt.xlabel('参数 p')
    plt.ylabel('对数似然 l(p)')
    plt.title(f'伯努利分布的对数似然函数 (n={n}, k={k})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 验证：MLE 估计就是样本均值
    print("数学推导验证:")
    print(f"  似然函数： L(p) = p^{k} * (1-p)^{n-k}")
    print(f"  对数似然： l(p) = {k}*log(p) + {n-k}*log(1-p)")
    print(f"  令 dl/dp = 0: {k}/p - {n-k}/(1-p) = 0")
    print(f"  解得： p̂ = {k}/{n} = {k/n:.4f}")
    ```

    从代码模拟结果可见，伯努利分布的 MLE 结果就是样本均值 $k/n$。与直觉完全一致，那似然函数和这套复杂流程有什么价值呢？首先，这是**证明而非猜测**，直觉告诉我们"8 正 2 反"对应 $p=0.8$，但推导证明了这个结论在任何情况下都成立；其次，这是通用方法，伯努利分布是最简单的例子，结论恰好直观，但其他分布的结论往往不那么简单（譬如下面案例 2 的正态分布的方差估计）；最后，推导过程揭示了 MLE 的本质 “ 令导数为零求解方式来最大化似然函数 ”，理解了这个原理才能判断何时该用 MLE、何时可能有更好的方法。

- **案例 2**（正态分布的 MLE）：与伯努利分布只需估计一个参数不同，正态分布有两个参数：均值 $\mu$ 和方差 $\sigma^2$。按照 MLE 的标准流程推导：
    首先，写出正态分布的似然函数：
    $$L(\mu,\sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$$
    然后，取对数得到：
    $$\ell(\mu,\sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2$$
    最后，分别对 $\mu$ 和 $\sigma^2$ 求导并令其为零，解得：

    - $\hat{\mu}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$（样本均值）
    - $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$（样本方差）

    以下代码将模拟生成正态分布数据，用这两个公式计算估计值，最后可视化真实分布与估计分布的差异。

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    # MLE 估计：正态分布
    true_mu, true_sigma = 5.0, 2.0
    n = 1000
    data = np.random.normal(true_mu, true_sigma, n)
    # MLE 估计
    mu_mle = np.mean(data)
    sigma2_mle = np.mean((data - mu_mle) ** 2)  # MLE 用 n 做分母
    sigma_mle = np.sqrt(sigma2_mle)

    print("=== 正态分布参数估计 ===")
    print(f"真实参数： μ = {true_mu}, σ = {true_sigma}")
    print(f"MLE 估计： μ̂ = {mu_mle:.4f}, σ̂ = {sigma_mle:.4f}")
    print()

    # 可视化
    x = np.linspace(true_mu - 4*true_sigma, true_mu + 4*true_sigma, 1000)
    def normal_pdf(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    plt.figure(figsize=(10, 6))

    # 真实分布
    plt.plot(x, normal_pdf(x, true_mu, true_sigma), 'g-', linewidth=2, label=f'真实分布： N({true_mu}, {true_sigma}²)')
    # MLE 估计分布
    plt.plot(x, normal_pdf(x, mu_mle, sigma_mle), 'r--', linewidth=2, label=f'MLE 估计： N({mu_mle:.2f}, {sigma_mle:.2f}²)')
    # 直方图
    plt.hist(data, bins=30, density=True, alpha=0.3, color='blue', edgecolor='black')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('正态分布的 MLE 估计')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    ```

### 最大后验估计（MAP）

**最大后验估计**（Maximum A Posteriori Estimation, MAP）是另一种点估计方法，其核心思想是**除了观测数据，还纳入我们对参数的先验知识（prior knowledge），找到让"后验概率最大"的参数值作为估计结果**。

举个直观的例子：假设你拿到一枚硬币，抛了 10 次，观察到 8 次正面、2 次反面。用 MLE 估计，你会得到 $p=0.8$。但如果你事先知道这枚硬币是正规赌场提供的（倾向于公平），那么你会认为 $p=0.8$ 这个估计太极端了，刚才 10 抛 8 正的数据很可能是一次小概率的偶然事件，真实的硬币不太可能这么不公平。这时你的"先验知识"告诉你硬币应该是公平的（$p$ 接近 0.5），而观测数据告诉你 $p$ 可能是 0.8。MAP 的做法是将两者结合起来：**后验概率 = 似然 × 先验**，选择让后验概率最大的参数值。结果会介于 0.5 和 0.8 之间，既尊重数据，又尊重先验。数学上，MAP 可表达为寻找使后验概率 $P(\theta|X)$ 最大的参数：

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|X)$$

根据[贝叶斯定理](probability-basics.md#贝叶斯定理)，后验概率可以展开为：

$$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}$$

由于 $P(X)$ 是实际数据本身的观察的结果，它与 $\theta$ 无关，在最大化时可以忽略掉，因此，MAP 等价于：

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(X|\theta)P(\theta)$$

即最大化**似然 × 先验**。与 MLE 相比，MAP 多了一个因子 $P(\theta)$，这就是先验知识的能够发挥作用的原因。

现在，已经学习了 MLE 和 MAP 两种估计方法，我们可以做一些比较，以便明确它们各自的使用场景：MLE 和 MAP 代表两种不同的统计哲学。当先验分布是均匀分布（即对所有参数值一视同仁）时，$P(\theta)$ 就是一个常数，此时 MAP 与 MLE 将给出相同的估计。当样本量很大，如趋于无穷时，数据的信息量远超先验，先验的影响被"淹没"（如观察到大量的实测数据都与先验不一致，那说明先验需要被修正了），此时 MAP 与 MLE 的估计结果也趋于一致。所以这些场景都没有必要选用 MAP，**MAP 的使用场景主要体现在数据较少、先验知识有价值的情况下**。

| 方法 | 优化目标 | 哲学立场 | 适用场景 |
|------|----------|----------|----------|
| MLE | 最大化似然 $P(X\|\theta)$ | 频率学派：参数是固定值，只相信数据 | 数据充足、无先验知识 |
| MAP | 最大化后验 $P(\theta\|X) = P(X\|\theta)P(\theta)$ | 贝叶斯学派：参数有先验分布，数据更新信念 | 数据较少、有先验知识 |


以下用代码来实例化抛硬币的案例，直观对比 MLE 和 MAP 的差异。假设真实硬币正面概率为 $p=0.7$（略微偏向正面），但我们只进行了 10 次抛掷（样本量较小）。代码使用 Beta(5,5) 作为先验分布。Beta 分布是描述概率值本身的概率分布，参数 $\alpha=5, \beta=5$ 表示我们相信硬币应该是公平的（$p$ 接近 0.5）。代码将计算 MLE 和 MAP 的估计结果，并通过可视化展示似然函数、先验分布和后验分布三者的关系。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# MLE vs MAP：抛硬币案例
# 假设我们怀疑硬币不正常，先验认为 p 接近 0.5

true_p = 0.7
n = 10  # 样本量较小，先验影响更明显
flips = np.random.binomial(1, true_p, n)
k = flips.sum()

# MLE 估计
p_mle = k / n

# MAP 估计（Beta 先验）
# 使用 Beta(α, β) 作为先验，等价于预先观测了 α-1 次正面和 β-1 次反面
alpha, beta = 5, 5  # 先验认为 p ≈ 0.5
p_map = (k + alpha - 1) / (n + alpha + beta - 2)

print(f"真实参数： p = {true_p}")
print(f"观测数据： n={n}, k={k}")
print(f"MLE 估计： p̂ = {p_mle:.4f}")
print(f"MAP 估计 (Beta({alpha},{beta}) 先验): p̂ = {p_map:.4f}")
print()

# 可视化
p_values = np.linspace(0.01, 0.99, 100)

# 似然函数
log_likelihood = k * np.log(p_values) + (n - k) * np.log(1 - p_values)

# 先验（Beta 分布）
from math import gamma as gamma_func
def beta_pdf(x, a, b):
    return (gamma_func(a + b) / (gamma_func(a) * gamma_func(b))) * (x ** (a-1)) * ((1-x) ** (b-1))

prior = np.array([beta_pdf(p, alpha, beta) for p in p_values])

# 后验（正比于似然 × 先验）
log_posterior = log_likelihood + np.log(prior + 1e-10)

# 归一化用于可视化
posterior = np.exp(log_posterior - log_posterior.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 似然
axes[0].plot(p_values, np.exp(log_likelihood - log_likelihood.max()), 'b-', linewidth=2)
axes[0].axvline(p_mle, color='r', linestyle='--', label=f'MLE: {p_mle:.2f}')
axes[0].set_xlabel('p')
axes[0].set_ylabel('似然（归一化）')
axes[0].set_title(f'似然函数')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 先验
axes[1].plot(p_values, prior, 'g-', linewidth=2)
axes[1].axvline(0.5, color='r', linestyle='--', label='先验均值： 0.5')
axes[1].set_xlabel('p')
axes[1].set_ylabel('先验密度')
axes[1].set_title(f'先验分布 Beta({alpha}, {beta})')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 后验
axes[2].plot(p_values, posterior, 'purple', linewidth=2)
axes[2].axvline(p_map, color='r', linestyle='--', label=f'MAP: {p_map:.2f}')
axes[2].axvline(true_p, color='g', linestyle=':', label=f'真实值： {true_p}')
axes[2].set_xlabel('p')
axes[2].set_ylabel('后验密度（归一化）')
axes[2].set_title('后验分布')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("关键洞察：")
print(f"  当样本量小时（n={n}），MAP 估计受到先验的强烈影响")
print(f"  MLE 可能过拟合数据（p̂={p_mle:.2f}），MAP 更稳健（p̂={p_map:.2f}）")
print(f"  当样本量大时，先验影响减弱，MLE 和 MAP 趋于一致")
```

### 无偏性与一致性

既然同一个参数可能有多种估计方法，那么如何评判哪个估计量更恰当呢？统计学提出了两个重要的评判标准：**无偏性**与**一致性**。

**无偏性**是指估计量的期望值正好等于真实参数值。通俗地说，如果我们重复抽样很多次，每次用这个估计方法计算一个估计值，这些估计值的平均值会收敛到真实参数，既不会系统性地偏高，也不会系统性地偏低。无偏性数学表达为：$E[\hat{\theta}] = \theta$。举个具体例子：用样本均值 $\bar{x}$ 估计总体均值 $\mu$。如果重复抽样 100 次，每次计算样本均值，这 100 个样本均值的平均值会非常接近 $\mu$，所以样本均值是总体均值的**无偏估计**。

但并非所有估计量都是无偏的。前面 [MLE 案例 2](#最大似然估计-mle) 中，正态分布的方差估计就是一个经典的有偏案例。因为计算样本方差时用的是样本均值 $\bar{x}$，而 $\bar{x}$ 比真实均值 $\mu$ 更"靠近"样本数据（毕竟 $\bar{x}$ 就是从这些数据算出来的）。数据离 $\bar{x}$ 更近，意味着 $(x_i - \bar{x})^2$ 系统性地小于 $(x_i - \mu)^2$，所以 MLE 低估了真实的方差。用 $n-1$ 做分母可以修正这个偏差 —— 分母变小，结果变大，正好抵消低估的部分。因此：

- 用分母 $n-1$ 的公式 $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$，得到的是**无偏估计**。
- 用分母 $n$ 的公式 $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$（即 MLE），得到的是**有偏估计**，系统性地低估方差。

**一致性**是指当样本量趋于无穷时，估计量会收敛到真实参数。通俗地说，数据越多，估计就越准，最终会"逼近"真实值。数学表达为：$\hat{\theta}_n \xrightarrow{p} \theta \quad \text{当} \quad n \to \infty$。MLE 通常满足一致性 —— 随着样本量增大，MLE 估计值会越来越接近真实参数。这也是为什么在大样本情况下，MLE 和 MAP 的结果趋于一致：数据量足够大时，数据的"声音"足够强，先验的影响自然减弱。

无偏性和一致性这两个标准看似相似，实则关注的角度不同。无偏性是对估计量"平均表现"的要求，即使只有少量数据，估计量的期望值也是正确的；一致性是对估计量"长期表现"的要求，需要大量数据才能保证估计接近真实值。两者是独立的标准：一个估计量可以无偏但不一致（如只用第一个观测值 $x_1$ 估计均值，期望正确但不会随数据增加而改进）；也可以有偏但一致（如 MLE 的方差估计，偏差会随样本量增加而减小）。在实践中，一致性往往更重要，因为只要样本足够大，有偏但一致的估计量偏差会逐渐消失，而无偏但不一致的估计量永远无法精确估计参数。


## 区间估计

点估计用一个具体数值作为参数的估计，简洁明了。但它有一个缺陷：**没有告诉我们这个估计有多可靠**。譬如，估计某城市居民平均收入为 5000 元，但这个估计是基于 10 人样本还是 10000 人样本得出的？如果是 10 人样本，估计值的可信度很低；如果是 10000 人样本，可信度就高得多，点估计只有一个数值结果，无法体现这种差异。

**区间估计**（Interval Estimation）解决了这个问题，它给出的不是一个数值，而是一个**范围**，同时附带一个**置信程度**。譬如说"平均收入在 4800 - 5200 元之间，置信水平 95%"，这就比单纯说"5000 元"更有信息量，既给出了估计结果，又量化了不确定性。

**置信区间**（Confidence Interval）是区间估计的具体形式。对于参数 $\theta$，置信区间是一个随机区间 $[L, U]$，满足：

$$P(L \leq \theta \leq U) = 1 - \alpha$$

其中 $\alpha$ 称为显著性水平，$1-\alpha$ 则称为**置信水平**（Confidence Level），实际操作中通常取 0.95 或 0.99。置信水平 95% 的真实含义不是"参数有 95% 概率落在这个区间内"，而是"如果重复抽样很多次，每次计算一个置信区间，大约 95% 的区间会包含真实参数"。这两种说法的差别在于**谁是随机的**：

- "参数有概率落在区间内"隐含参数是随机变量，区间是固定的 —— 这是贝叶斯学派的观点
- "95%的区间包含参数"则认为参数是固定的值，区间是随机的 —— 这是频率学派的观点

举个具体例子：假设真实均值 $\mu=50$。某次抽样算出置信区间 $[48, 52]$。在频率学派看来，$\mu$ 是固定的 50，区间 $[48, 52]$ 要么包含 50（概率=1），要么不包含（概率=0），不存在"95%概率包含"的说法。95% 的含义是：如果重复抽样 100 次，大约 95 次算出的区间会包含 $\mu=50$，大约 5 次不包含。这是对"这个估计方法"的长期表现评价，不是对"某个具体区间"的概率描述。每个具体的置信区间一旦算出来，它的"命运"已经确定，要么包含，要么不包含，没有概率可言。

以下代码通过模拟实验来直观展示置信区间的含义：设定真实均值 $\mu=100$，进行 50 次独立抽样，每次计算一个 95% 置信区间，观察有多少个区间包含真实值。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 置信区间演示
true_mu = 100
true_sigma = 15
n = 30
n_experiments = 50

# 模拟多次抽样，计算置信区间
intervals = []
contains_true = []

for i in range(n_experiments):
    sample = np.random.normal(true_mu, true_sigma, n)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)  # 无偏估计
    
    # 95% 置信区间
    # 使用 t 分布近似（样本量较大时可用正态分布）
    from math import sqrt
    margin = 1.96 * sample_std / sqrt(n)  # 简化用正态分布
    lower = sample_mean - margin
    upper = sample_mean + margin
    
    intervals.append((lower, upper))
    contains_true.append(lower <= true_mu <= upper)

# 可视化
plt.figure(figsize=(12, 8))
for i, ((lower, upper), contains) in enumerate(zip(intervals, contains_true)):
    color = 'green' if contains else 'red'
    plt.plot([lower, upper], [i, i], color=color, linewidth=1.5)

plt.axvline(true_mu, color='blue', linestyle='--', linewidth=2, label=f'真实均值 μ={true_mu}')
plt.xlabel('均值')
plt.ylabel('实验序号')
plt.title(f'95% 置信区间 ({sum(contains_true)}/{n_experiments} 包含真实值)')
plt.legend()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
plt.close()
```

**标准误差**（Standard Error, SE）是一个与置信水平可相互转化的、用来描述置信区间不确定性的概念。它反映了置信区间估计量的"波动程度"，置信区间 = 点估计 ± 临界值 × 标准误差。譬如，95% 置信水平下，样本均值的置信区间为 $\bar{x} \pm 1.96 \times SE(\bar{x})$。（1.96 是标准正态分布对应 95% 置信水平的临界值），$SE(\bar{x})$ 就是标准误差。从这个公式可以看出**标准误差越大，置信区间越宽**（估计不确定性大）；**标准误差越小，置信区间越窄**（估计更精确）。

标准误差与标准差这两个概念名称和作用都有相似之处，但它们关注的角度不同：标准差衡量原始数据的离散程度，描述的是数据本身；标准误差衡量估计量的离散程度，描述的是估计结果。譬如，从某城市抽取 100 人调查收入，样本均值是 5000 元。这 100 人的收入数据有高有低，标准差描述的是这 100 人收入的离散程度；如果我们再抽 100 人、再抽 100 人，重复 100 次，会得到 100 个不同的样本均值，这些样本均值也有高有低，标准误差描述的就是这些样本均值的离散程度。

对于样本均值，标准误差为 $SE(\bar{x}) = \frac{\sigma}{\sqrt{n}}$，这个公式说明**样本量越大，标准误差越小**。从数学上很容易解释，$n$ 在分母中，意味着样本量翻倍，标准误差缩小为原来的 $\frac{1}{\sqrt{2}} \approx 0.71$。这与直觉一致，数据越多，估计越稳定，波动越小。实际应用中，总体标准差 $\sigma$ 通常未知，需要用样本标准差 $s$ 来估计：$\hat{SE}(\bar{x}) = \frac{s}{\sqrt{n}}$，这就是为什么在计算置信区间时，公式中的 $\frac{\sigma}{\sqrt{n}}$ 实际上用的是 $\frac{s}{\sqrt{n}}$。

## 假设检验

前文介绍了两类估计方法：点估计给出参数的具体数值，区间估计给出参数的可能范围。这两种方法都是"主动探索"，已经有观测统计到的数据，想知道参数是什么。在实际应用中，还经常遇到另一类"被动判断"的问题：验证某个关于参数的假设是否成立。譬如，新算法宣称比旧算法提升 10%，这个声称可信吗？某特征与目标变量相关，这是真实的规律还是数据的偶然波动？A/B 测试中，新方案的表现更好，是真正的改进还是随机误差？这类问题需要一套系统的方法来判断假设的合理性，这就是**假设检验**（Hypothesis Testing）要解决的问题。

假设检验的思路可以类比于"法庭审判"：我们先假定被告无罪（原假设），然后根据证据判断是否有足够理由推翻这个假定。证据越充分（数据越极端），推翻无罪假定的理由就越强。这个类比强调了假设检验的两个关键特点：第一，不直接证明假设"成立"，而是判断是否有足够证据"拒绝"它；第二，结论永远是"拒绝"或"不能拒绝"，不存在"接受"的说法，因为即使不能拒绝原假设，只意味着证据不足，不代表原假设一定正确。

还是用抛硬币为例来介绍设检验的工作流程：拿到一枚硬币，想判断它是否公平。抛了 100 次，观察到 65 次正面、35 次反面。这个结果与"公平硬币应出现约 50 次正面"的预期相差较大，但偏差多大才算不公平？可能是偶然因素可能导致偏离，也可能是硬币本身有问题。你可以通过以下步骤来量化判断：

1. **提出假设**。原假设 $H_0$："硬币公平，正面概率 $p=0.5$"；备择假设 $H_1$："硬币不公平，$p \neq 0.5$"。注意，$H_0$ 是保守的"无罪推定"，$H_1$ 是我们想验证的"有罪指控"。
2. **选择显著性水平**。设定 $\alpha = 0.05$，意思是：只有当数据极端到"在公平硬币下出现概率低于 5%"的程度，才愿意拒绝 $H_0$。这相当于法庭的"证据门槛"。
3. **计算检验统计量**。正面次数 $k=65$ 就是检验统计量，它直接反映了数据与原假设的偏离程度。
4. **计算 p 值**。如果硬币真的公平（$p=0.5$），出现 65 次或更多正面（以及对称的 35 次或更少正面）的概率是多少？这个概率就是 p 值，约 0.0035（具体计算见后续代码）。
5. **做出判断**。p = 0.0035 < α = 0.05，拒绝 $H_0$。结论：有统计显著性证据认为硬币不公平。

这套流程中最为关键，也最容易误解的概念是 **p 值**。p 值在原假设为真的条件下，观测到当前数据或更极端数据的概率。用程序员熟悉的类比来理解 p 值：想象你在测试一个声称"公平"的随机数生成器，它应该均匀生成 0-9。你跑了一次测试，发现生成了 100 个数字，其中 90 个都是 9。如果生成器真的公平，出现这种情况的概率极低（p 值极小）。这个 p 值不是"生成器不公平的概率"，而是"如果生成器公平，观察到这种极端结果的惊讶程度"。

所以，p 值 **不是**原假设为真的概率，而是如果原假设为真，观察到这种结果的惊讶程度。p 值小，说明惊讶程度高，数据与原假设严重不符，因此有理由怀疑原假设。这个理解是假设检验的核心思想：我们永远不知道原假设是否真的正确，只知道当前数据与原假设是否"吻合"。p 值小意味着数据很极端，在原假设下不太可能出现，这时我们有理由怀疑原假设，同时也必须说明，"怀疑"不等于"证伪"，极端数据也可能是偶然事件，只是概率很低罢了。

假设检验为机器学习中的特征选择（检验某特征是否与目标变量相关）、模型比较（检验模型 A 是否显著优于模型 B）等问题提供了科学的、可量化、可操作的依据。我们经常调侃训练模型是炼丹，充满了运气与随机，但同时也必须承认，训练模型是科学的和工程化的，灵光一闪固然重要，但并不能靠着碰运气来筛选出一个个正确或错误的想法。下面这段程序对假设检验进行了可视化模拟。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 假设检验演示：硬币是否公平
# H0: p = 0.5 (硬币公平)
# H1: p ≠ 0.5 (硬币不公平)
n = 100
flips = np.random.binomial(1, 0.65, n)  # 实际 p = 0.65（不公平）
k = flips.sum()

# 计算 p 值（双尾检验）
# 在 H0 下，k ~ Binomial(n, 0.5)
from math import comb

def binomial_pmf(k, n, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# 计算观察到 k 或更极端情况的概率
p_value = 0
expected = n * 0.5

for i in range(n + 1):
    if abs(i - expected) >= abs(k - expected):  # 比 k 更极端
        p_value += binomial_pmf(i, n, 0.5)

# 可视化
k_values = np.arange(0, n + 1)
pmf = [binomial_pmf(i, n, 0.5) for i in k_values]

plt.figure(figsize=(12, 5))

# 绘制分布
plt.bar(k_values, pmf, color='steelblue', edgecolor='black', alpha=0.7, label='H0 下的分布')

# 标记极端区域
extreme_mask = np.array([abs(i - expected) >= abs(k - expected) for i in k_values])
extreme_k = k_values[extreme_mask]
extreme_pmf = np.array(pmf)[extreme_mask]
plt.bar(extreme_k, extreme_pmf, color='red', edgecolor='black', alpha=0.7, label=f'极端区域 (p 值)')

# 标记观测值
plt.axvline(k, color='orange', linestyle='--', linewidth=2, label=f'观测值 k={k}')
plt.axvline(expected, color='green', linestyle=':', linewidth=2, label=f'期望值 E[k]={expected}')

plt.xlabel('正面次数 k')
plt.ylabel('概率')
plt.title(f'假设检验：硬币是否公平 (n={n})')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.xlim(30, 70)
plt.tight_layout()
plt.show()
plt.close()

print("=== 假设检验结果 ===")
print(f"H0: 硬币公平 (p = 0.5)")
print(f"观测数据： n = {n}, k = {k} (正面比例 = {k/n:.2%})")
print(f"p 值： {p_value:.4f}")
print(f"显著性水平 α = 0.05")
print()
if p_value < 0.05:
    print(f"结论： p 值 ({p_value:.4f}) < α (0.05)，拒绝 H0")
    print("      有统计显著性证据认为硬币不公平")
else:
    print(f"结论： p 值 ({p_value:.4f}) ≥ α (0.05)，不能拒绝 H0")
    print("      没有足够证据认为硬币不公平")
```

## 本章小结

本章介绍了统计推断的核心方法，它们构成了从数据到结论的完整链条：

**点估计**解决"参数是什么"的问题。MLE 只看数据，选择最能解释观测结果的参数；MAP 结合先验知识，在数据不足时提供更稳健的估计。两种方法代表频率学派和贝叶斯学派的不同哲学，但在大样本情况下趋于一致。

**区间估计**解决"估计有多可靠"的问题。置信区间给出参数的可能范围，置信水平量化了这个范围"包含真实参数"的概率（频率学派视角）或"参数落在此范围"的概率（贝叶斯视角）。标准误差描述估计量的波动程度，样本量越大，估计越精确。

**假设检验**解决"假设是否成立"的问题。通过 p 值量化数据与假设的吻合程度，p 值小意味着数据在假设下很极端，有理由怀疑假设。假设检验只能"拒绝"或"不能拒绝"，不存在"接受"的说法。

这三类方法在机器学习中无处不在：模型训练本质上是参数估计（MLE 或 MAP），正则化可解释为引入先验的 MAP 估计，模型评估涉及置信区间和假设检验。有了统计推断的作为基础，后面才能理解机器学习为什么"有效"、何时可能"失效"。



## 练习题

1. 为什么 MLE 的方差估计是有偏的？为什么除以 $n-1$ 才是无偏的？
   <details>
   <summary>参考答案</summary>

   当我们用样本均值 $\bar{x}$ 估计 $\mu$ 时，样本数据相对于 $\bar{x}$ 比 $\mu$ 更"近" —— 因为 $\bar{x}$ 就是从这些数据计算出来的。这导致 $\sum(x_i - \bar{x})^2$ 系统性地小于 $\sum(x_i - \mu)^2$。

   数学上，可以证明 $E[\sum(x_i - \bar{x})^2] = (n-1)\sigma^2$，所以除以 $n-1$ 才能得到无偏估计。

   直观理解：计算样本均值用掉了一个"自由度"，剩下的自由度是 $n-1$。

   </details>

2. 在什么情况下 MLE 和 MAP 会给出相同的估计？
   <details>
   <summary>参考答案</summary>

   当先验是均匀分布（无信息先验）时，MLE 和 MAP 给出相同的估计。因为此时 $P(\theta)$ 是常数，最大化后验等价于最大化似然。

   另一种情况是样本量趋于无穷时，先验的影响被数据"淹没"，MLE 和 MAP 趋于一致。

   </details>

3. 解释为什么"95% 置信区间"不意味着"参数有 95% 概率落在这个区间"。
   <details>
   <summary>参考答案</summary>

   在频率学派的框架下，参数是固定值，不是随机变量。因此不能说"参数有某个概率落在某处"。

   95% 置信区间的正确理解是：如果我们重复抽样并计算置信区间，大约 95% 的区间会包含真实参数。这是关于"方法"的置信度，而不是"参数"的概率。

   在贝叶斯框架下，参数被视为随机变量，"95% 可信区间"确实意味着参数有 95% 概率落在这个区间。

   </details>