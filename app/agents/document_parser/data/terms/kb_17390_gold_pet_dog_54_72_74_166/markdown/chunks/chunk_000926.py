from langchain_core.documents import Document

chunk = Document(
    page_content=('일 때\n'
 '다) 간장의 3/4 이상을 잘라내었을 때\n'
 '라) 양쪽 고환 또는 양쪽 난소를 모두 잃었을 때\n'
 '‘흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때’라 함은# 4)- \n'
 '아래의 경우 중 하나에 해당하는 때를 말한다.\n'
 '가) 한쪽 폐 또는 한쪽 신장을 전부 잘라내었을 때\n'
 '나) 방광 기능상실로 영구적인 요도루, 방광루, 요관 장문합 상태\n'
 '다) 위, 췌장을 50% 이상 잘라내었을 때\n'
 '라) 대장절제, 항문 괄약근 등의 기능장해로 영구적으로 장루, 인공항\n'
 '문을 설치한 경우(치료과정에서 일시적으로 발생하는 경우는 제외)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
