from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나이가 증가하는 것으로 합니다.\n'
 '- ③ 피보험자의 나이 또는 성별에 관한 기재사항이 사실과 다른 경우에는 정정된 나이 또\n'
 '- 는 성별에 해당하는 보험금 및 보험료로 변경합니다.\n'
 '<예시안내>[보험나이 계산]\n'
 '생년월일 : 1988년 10월 2일예1) 계 약 일 : 2022년 3월 13일⇒ 2022년 3월 13일\n'
 '- 1988년 10월 2일\n'
 '33년 5개월 11일 = 33세예 2) 계 약 일 : 2022년 4월 13일\n'
 '⇒ 2022년 4월 13일\n'
 '- 1988년 10월 2일\n'
 '33년 6개월 11일 = 34세[계약해당일 계산]'),
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
