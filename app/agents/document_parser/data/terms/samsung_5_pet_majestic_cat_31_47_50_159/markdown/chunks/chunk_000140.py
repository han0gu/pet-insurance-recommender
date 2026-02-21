from langchain_core.documents import Document

chunk = Document(
    page_content=('회사와 계약자가 합의하여 관할법원을 달리 정할 수 있습니다.# 제 41조 (소멸시효)보험금청구권, 만기환급금청구권, 보험료 반환청구권, '
 '해약환급금 청구권, 계약자적립액\n'
 '및 미경과보험료 반환청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다.<용어풀이>\n'
 '[소멸시효]\n'
 '소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합니다. 보험금 지급사유가 2021년 4월 1일- 45 -에 발생하였음에도 2024년 '
 '4월 1일까지 보험금을 청구하지 않는 경우 소멸시효가 완성되어 보험'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
