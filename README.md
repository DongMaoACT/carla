
# ADAS application based on Carla simulator

#### 引用

CARLA Simulator：https://github.com/carla-simulator/carla

CARLA Documentation：https://carla.readthedocs.io/en/latest/

项目介绍

**本项目旨在探究不同人车交互方式的ADAS应用在使用方面的正反馈性（能够为驾驶员提供有效信息并提升驾驶感受），进而设计了力反馈装置震动、语音提示、2D雷达界面、环境感知的3D场景四种不同的反馈模式。并设计交通场景（如通过无信号的十字交叉路口），通过招募式的实验测试，采集受测者的力反馈装置操纵数据和脑电波数据，通过数据集的相关性分析，判断不同反馈模式的ADAS应用的正反馈性。**

#### 功能预览

#### 实时场景

<img src="Src\场景.png" alt="场景" style="zoom: 33%;" />

#### 2D雷达界面
<div align=center>
<img src="Src\2D.png" alt="2D" align="center" style="zoom: 50%;" />
</div>

#### 3D环境感知(该图是被监察视角)
<div align=center>
<img src="Src\3DUI.png" alt="3DUI" align="center" style="zoom: 67%;" />
</div>

#### 项目搭建
​		**直接拉取本地进行项目搭建，内包含了CARLA Simulator，但是CALRA特别版的Unreal 4.26 引擎需要自行编译构建。**（PS：因为部分Sever端的代码被修改过，不能直接拉取官方代码）

​		Carla项目搭建：https://carla.readthedocs.io/en/latest/build_windows/

​		按照Make工具进行构建完成之后，建议make package打包项目，然后运行Sever。

​		通过pyhton运行./PyhtonAPI/examples/3d_adas.py 和 ./PyhtonAPI/examples/manual_control.py进行实验。

​		**PS:以上所有命令行操纵均在项目根目录下进行**

#### 初期实验

​		**实验数据初步分析如下：**

<img src="Src\ADASEEG.png" alt="ADAS" style="zoom: 67%;" />

#### 实验架构

本项目基于CARLA的C/S异步架构，主要代码在于Client端的修改（回调、代理模式），服务端只修改了语义分割相机的内容（取巧了场景的简化部分）。

<img src="Src\系统架构.png" alt="系统" style="zoom: 67%;" />
