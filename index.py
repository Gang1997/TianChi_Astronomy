# Created by [Yuexiong Ding] at 2018/2/25
# 分割label
#

import numpy as np

Labels = np.loadtxt(r'D:\MyProjects\TianChi\Astronomy\DataSet\NormalizedTrainingData\Index\EncodeType\1.txt',
                    delimiter=',')
np.savetxt('./DataSet/5D/Labels/TrainingLabels.txt', Labels[:int(50000 * 0.7), :], delimiter=',')
np.savetxt('./DataSet/5D/Labels/DevLabels.txt', Labels[int(50000 * 0.7):, :], delimiter=',')
