CharRecog
=========

�j���[�����l�b�g���g���������F���G���W���ł��B
A character recognition engine using neural networks.


�������
--------
OS �� Windows 7 �܂��͂���ȏ�� 64 �r�b�g�ł��K�v�ł��B
�i�\�[�X����� 32 �r�b�g�ł��r���h���邱�Ƃ��ł��܂��B�j


�܂��������Ă݂������ցiMNIST�摜�Z�b�g�̊w�K�j
----------------------------------------

1. GitHub ���烍�[�J���Ƀ��W���[���ꎮ���_�E�����[�h���܂��B

    git clone https://github.com/kensak/CharRecog.git

2. [THE MNIST DATABASE of handwritten digits �̃T�C�g](http://yann.lecun.com/exdb/mnist/)
���� 4 �̃f�[�^�t�@�C�����_�E�����[�h���A�𓀂��Ă��� data\MNIST �t�H���_�[�ɒu���܂��B

3. �����t�H���_�[�ɂ��� run.bat �����s����ƁAMNIST_test_data �� MNIST_train_data �Ƃ����t�H���_�[���ł��A
png �`���̕����摜���������܂�܂��B

4. ���[�g�ɂ��� demo-MNIST.bat �����s����Ɗw�K���n�܂�܂��B
200 ���[�v���s���A10 ���[�v���ƂɊw�K�Z�b�g�ƕ]���Z�b�g�ł̔F�������v�Z���܂��B
�w�K���I������ƁA3 �̃p�[�Z�v�g�����E���C���[�̏d�ݏ����摜�ɂ��ďo�͂��܂��B
�l�b�g���[�N�̏��� NN_params.bin �ɏo�͂���A�����F���Ɏg�p�ł��܂��B  
�Ⴆ�΁A

    bin64\CharRecog.exe -v -b -h 28 -w 28 TEST data\MNIST\MNIST_test_data

�����s����ƁA�]���Z�b�g�̔F�������Ăьv�Z���܂��B

 
�p�����[�^�̉��
----------------
���X�g�̊Ԃɋ�s�����ނƁA���ꂼ��̃��X�g�� `<p>` �^�O���}������A�s�Ԃ�
�L���Ȃ�܂��B
 
    def MyFunction(param1, param2, ...)
 
+   `param1` :
    _�p�����[�^1_ �̐���
 
+   `param2` :
    _�p�����[�^2_ �̐���
 
�֘A���
--------
### �����N�A�l�X�g�������X�g
1. [�����N1](http://example.com/ "�����N�̃^�C�g��")
    * ![�摜1](http://github.com/unicorn.png "�摜�̃^�C�g��")
2. [�����N2][link]
    - [![�摜2][image]](https://github.com/)
 
  [link]: http://example.com/ "�C���f�b�N�X�^�̃����N"
  [image]: http://github.com/github.png "�C���f�b�N�X�^�̉摜"
 
### ���p�A�l�X�g�������p
> ����͈��p�ł��B
>
> > �X�y�[�X������� `>` ���d�˂�ƁA���p�̒��ň��p���ł��܂����A
> > GitHub�̏ꍇ�A1�s�O�ɋ�̈��p�������ƁA�������}�[�N�A�b�v����܂���B
 
���C�Z���X
--------
Dual licensed under the [MIT license][MIT] and [GPL license][GPL].
[MIT]: http://www.opensource.org/licenses/mit-license.php
[GPL]: http://www.gnu.org/licenses/gpl.html