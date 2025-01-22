# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

from utils import evaluate_places

prediction = ["London"] * 500
dev_path = "birth_dev.tsv"
total, correct = evaluate_places(dev_path, prediction)
print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))

# Correct: 25.0 out of 500.0: 5.0%