from langchain_core.documents import Document

chunk = Document(
    page_content=('- 호에 해당하는 보험·공제로서 보험·공제 계약 또는 보험료·공제료 납입영수증에 장애인전용 보험·공\n'
 '- 제로 표시된 보험·공제의 보험료·공제료를 말한다.\n'
 '- ② 소득세법 제59조의4 제1항 제2호에서 "대통령령으로 정하는 보험료"란 다음 각 호의 어느 하나에\n'
 '- 해당하는 보험·보증·공제의 보험료·보증료·공제료 중 기획재정부령으로 정하는 것을 말한다.\n'
 '- 1. 생명보험\n'
 '- 2. 상해보험\n'
 '- 3. 화재·도난이나 그 밖의 손해를 담보하는 가계에 관한 손해보험'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
