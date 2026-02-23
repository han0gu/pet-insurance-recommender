from langchain_core.documents import Document

chunk = Document(
    page_content=('자가 합의하여 관할법원을 달리 정할 수 있습니다.# 제33조(소멸시효)보험금청구권, 보험료 또는 환급금 반환청구권은 3년간 행사하지 '
 '않으면 소멸시효가 완성됩니다.【소멸시효】 일정기간 행사하지 않으면 권리를 소멸시키는 제도입니다. 소멸시효는 권리를 행사할 수 있는\n'
 '때로부터 진행합니다.당신에게 좋은보험 삼성화재- 18 -【예시】 보험금 지급사유가 2022년 1월 1일에 발생하였음에도 2025년 1월 '
 '1일까지 보험금을 청구하지 않는'),
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
