import json
import pdb

file_path = "/mnt/hdd/zhurui/code/Truly-Visual-CoT/outputs_refocus/mathvision/qwen2_5/extracted_answer.json"

with open(file_path, "r") as f:
    data = json.load(f)

result = {}
total_num = len(data)
num_correct_old = 0
num_correct_new = 0
for key, item in data.items():
    if item["true_false"] == True:
        num_correct_old += 1
    if item["refocus_true_false"] == True:
        num_correct_new += 1
    if item["true_false"] != item["refocus_true_false"]:
        result[key] = item
# 保存结果
with open("result_analysis.json", "w") as f:
    print(len(result))
    num_true_to_false = 0
    num_false_to_true = 0
    for item in result.values():
        if item["true_false"] == True and item["refocus_true_false"] == False:
            num_true_to_false += 1
        else:
            num_false_to_true += 1
    json.dump(result, f, indent=4)

print(f"true to false: {num_true_to_false}, false to true: {num_false_to_true}")
print(f"old correct: {num_correct_old}, new correct: {num_correct_new}")
print(
    f"old accuracy: {num_correct_old / total_num * 100:.2f}%, new accuracy: {num_correct_new / total_num * 100:.2f}%"
)
