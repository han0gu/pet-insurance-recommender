from langchain_core.documents import Document

chunk = Document(
    page_content=('- 종이 정정되기 이전에는 「나이 및 품종이 정정되기 전에 적용된 보험료율」 의 「나\n'
 '- 이 및 품종이 정정된 후에 적용해야할 보험료율」 에 대한 비율에 따라 보험금을 삭감\n'
 '- 하여 지급합니다.\n'
 '<예시안내># [계약해당일 계산]최초계약일과 동일한 월, 일을 말합니다.\n'
 '계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일\n'
 '단, 계약해당일 2월 29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.# 제19조 (특별약관의 소멸)① 보험증권에 기재된 '
 '반려묘가 보험기간 중에 사망하여 보험의 목적에 대해 이 특별약'),
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
