from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1988년 10월 2일\n'
 '33년 5개월 11일 = 33세- \n'
 '# [계약해당일 계산]최초계약일과 동일한 월, 일을 말합니다.\n'
 '계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일\n'
 '단 , 계약해당일 2월 29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.| 예 2) 계 약 일 : 2022년 4월 13일 ⇒ '
 '2022년 4월 13일 - 1988년 10월 2일 33년 6개월 11일 = 34세 |\n'
 '| --- |\n'
 '-'),
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
