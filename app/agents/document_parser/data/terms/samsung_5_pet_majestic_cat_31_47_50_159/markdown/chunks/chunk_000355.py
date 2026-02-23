from langchain_core.documents import Document

chunk = Document(
    page_content=('에 성형수술이 가능하다는 진단을 받은 경우에는 그 진단으로 대체할 수 있습니다)을\n'
 '받은 경우 아래에 정한 금액을 상해흉터복원(성형) 수술비로 보험수익자에게 지급합니\n'
 '다.| 구 분 | 안면부 | 상지·하지 |\n'
 '| --- | --- | --- |\n'
 '| 지급액 | 수술 1cm당 14만원 | 수술 1cm당 7만원 (단, 3cm이상의 경우에 한합니다) |\n'
 '- ② 제1항에서 길이측정이 불가한 피부이식수술 등의 경우 수술 cm는 최장직경으로 합니\n'
 '- 다.\n'
 '- ③ 제1항에서 정한 상해흉터복원(성형) 수술비는 하나의 사고에 대하여 500만원을 한도'),
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
