CharRecog
=========
�j���[�����l�b�g���g���������F���G���W���ł��B  
A character recognition engine using a neural network.

�������
--------
OS : Windows 7 �܂��͂���ȏ�A64 �r�b�g��  
�i�\�[�X����� 32 �r�b�g�ł��r���h���邱�Ƃ��ł��܂��B�j

RAM : 8GB �ȏ�

�܂��������Ă݂������ցiMNIST�摜�Z�b�g�̊w�K�j
----------------------------------------
1. GitHub ���烍�[�J���Ƀ��W���[���ꎮ���_�E�����[�h���܂��B
```
git clone https://github.com/kensak/CharRecog.git
```

2. [THE MNIST DATABASE of handwritten digits �̃T�C�g](http://yann.lecun.com/exdb/mnist/)
���� 4 �̃f�[�^�t�@�C�����_�E�����[�h���A�𓀂��Ă��� data\MNIST �t�H���_�[�ɒu���܂��B

3. �����t�H���_�[�ɂ��� run.bat �����s����ƁAMNIST_test_data �� MNIST_train_data �Ƃ����t�H���_�[���ł��A
png �`���̕����摜���������܂�܂��B

4. ���[�g�ɂ��� demo-MNIST.bat �����s����Ɗw�K���n�܂�܂��B
200 ���[�v���s���A10 ���[�v���ƂɊw�K�Z�b�g�ƕ]���Z�b�g�ł̔F�������v�Z���܂��B
�w�K���I������ƁA3 �̃p�[�Z�v�g�����E���C���[�̏d�ݏ����摜�ɂ��ďo�͂��܂��B
�l�b�g���[�N�̏��� NN_params.bin �ɏo�͂���A�����F���Ɏg�p�ł��܂��B  
�Ⴆ�΁A
```
bin64\CharRecog.exe -v -b -h 28 -w 28 TEST data\MNIST\MNIST_test_data
```
�����s����ƁA�]���Z�b�g�̔F�������Ăьv�Z���܂��B

����
----
+ ���񏈗� : �C���e�� TBB�AOpenMP, C++ AMP �Ȃǂ��g���A�}���`�R�A CPU �� GPU �ɂ����񏈗��������Ȃ��Ă��܂��B
+ �s��v�Z�ɂ�鍂���� : ���ׂẴT���v���摜����������ہA�ł��邾�����[�v���񂳂����̍s��v�Z�ɂ�菈���������Ȃ��܂��B
+ ���܂��܂Ȏ�@�ւ̑Ή� : autoencoding, convolutional layer, max pooling, maxout, dropout �̊e��@��
  ���͉摜�̃����_���� affine �ϊ��ȂǂɑΉ����Ă��܂��B
+ �r���h�ςݎ��s�t�@�C���ł� float �Ōv�Z�������Ȃ��܂����A�\�[�X���ꃖ���ύX���ă��r���h����� double �ɂ��Ή����܂��B
+ �w�K���ꂽ�d�݂��摜�Ƃ��ďo�͂ł��܂��B
 
�p�����[�^�̉��
--------------
doc �t�H���_�[�ɂ���}�j���A�����������������B
 
 �r���h�̕��@
-----------
src\ReadMe.txt ���������������B

���C�Z���X
--------
Dual licensed under the [MIT license][MIT] and [GPL v2 license][GPL].

[MIT]: http://www.opensource.org/licenses/mit-license.php
[GPL]: http://www.gnu.org/licenses/gpl.html