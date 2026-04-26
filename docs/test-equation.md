# 公式编号语法测试

本文档用于测试新的公式编号和引用语法。

## 单公式编号测试

使用 `$$[label] 公式 $$` 语法：

$$[eq:test1] E = mc^2 $$

这是著名的爱因斯坦质能方程，我们将在后面引用它。

## 另一个单公式

$$[eq:test2] F = ma $$

牛顿第二定律，力等于质量乘以加速度。

## 多公式共用编号测试

使用 `$$$[label] ... $$$` 语法将多个公式放在同一个编号下：

$$$[eq:multi]
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla L_t)^2$$
$$$

以上是 Adam 优化器的一阶矩和二阶矩更新公式。

## 公式引用测试

使用 `{{label}}` 语法引用公式：

- 质能方程见公式 {{eq:test1}}
- 牛顿第二定律见公式 {{eq:test2}}
- Adam 的两个矩更新公式见公式 {{eq:multi}}

## 混合测试

以下测试引用和普通公式的混合：

这是一个不带编号的普通公式：

$$ \int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2} $$

这是一个带编号的公式：

$$[eq:gaussian] \mathcal{N}(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$

高斯分布公式 {{eq:gaussian}} 是概率统计中最重要的分布之一。

## 兼容旧语法测试

以下测试旧的 HTML 注释语法是否仍然有效：

$$[eq:legacy] a^2 + b^2 = c^2 $$

勾股定理见公式 {{eq:legacy}}。

以及 LaTeX 语法：

\begin{equation}
\label{eq:latex}
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
\end{equation}

麦克斯韦方程组中的电场散度方程见公式 \eqref{eq:latex}。