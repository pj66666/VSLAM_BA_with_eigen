# 1 实际效果
具体见demo视频
![image](https://github.com/pj66666/VSLAM_BA_with_eigen/assets/68932539/7c487517-6f84-418a-bd29-98d9be2c57c8)

# 2 问题
利用舒尔补去加速线性方程求解时，发现每次计算非常耗时，完全不能实时。然后采用了两种方法去解决这个问题，一个是利用LDLT分解(基于eigen);另一个是基于共轭梯度法，详见代码

除此以外，利用openMP在构建海塞矩阵H时候进行加速。


# 编译


在代码中选择其中一种去运行，第一种基于g2o，第二种基于eigen

// Optimize(active_kfs, active_landmarks);

Optimize_only_with_eigen(active_kfs, active_landmarks);


[博客](https://blog.csdn.net/qq_49561752/category_12514411.html?spm=1001.2014.3001.5482)
