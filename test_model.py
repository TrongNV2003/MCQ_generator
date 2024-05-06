from questiongenerator import QuestionGenerator
from questiongenerator import print_qa

qg = QuestionGenerator()

with open('articles/philosophy.txt', 'r',encoding='utf-8') as a:
    article = a.read()

qa_list = qg.generate(
    article,
    num_questions=5,
    answer_style='multiple_choice'
)
# print_qa(qa_list, show_answers=False)
print_qa(qa_list, show_answers=True)