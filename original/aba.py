text = "53-year-old white female with past medical history of coronary artery disease status post stenting, history of mi, type 1 diabetes, presents to the emergency department with complaints of midsternal chest pain side in onset 2 hours prior to arrival to the ed. patient describes aching discomfort in the midsternal region with radiation to left shoulder blade. she reports associated nausea, diaphoresis and one episode of vomiting. patient describes symptoms consistent with her prior heart attack. patient denies associated abdominal pain, no reported fevers or chills. denies headache or neck stiffness. eyes dizziness. denies urinary complaints. patient does report her blood sugars have been elevated recently."
# 定义位置和长度
positions = [(479, 5), (232, 8)]

# 提取这些位置的词
for start, length in positions:
    print(f"Position {start}-{start+length}: {text[start:start+length]}")
