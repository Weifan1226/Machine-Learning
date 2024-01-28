import matplotlib.pyplot as plt
from IPython import display

plt.ion() 

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0) # 最小值为0
    plt.text(len(scores)-1, scores[-1], str(scores[-1])) # 显示最后一个得分
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1])) # 显示平均分
    plt.show(block=False) # 不阻塞
    plt.pause(.1)