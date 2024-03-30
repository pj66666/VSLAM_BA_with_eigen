实际效果
![image](https://github.com/pj66666/VSLAM_BA_with_eigen/assets/68932539/7c487517-6f84-418a-bd29-98d9be2c57c8)

问题：利用舒尔补去加速线性方程求解时，发现每次计算非常耗时，完全不能实时。然后采用了两种方法去解决这个问题，一个是利用LDLT分解(基于eigen);另一个是基于共轭梯度法，详见代码
