from langchain_core.documents import Document

chunk = Document(
    page_content=('- 8) "뚜렷한 시야 장해" 라 함은 한 눈의 시야 범위가 정상시야 범위의 60% 이하\n'
 '- 로 제한된 경우를 말한다. 이 경우 시야검사는 공인된 시야검사방법으로 측정\n'
 '- 하며, 시야장해 평가 시 자동시야검사계(골드만 시야검사)를 이용하여 8방향\n'
 '- 시야범위 합계를 정상범위와 비교하여 평가한다.\n'
 '- 9) "눈꺼풀에 뚜렷한 결손을 남긴 때" 라 함은 눈꺼풀의 결손으로 눈을 감았을 때\n'
 '- 각막(검은자위)이 완전히 덮이지 않는 경우를 말한다.\n'
 '- 10) "눈꺼풀에 뚜렷한 운동장해를 남긴 때" 라 함은 눈을 떴을 때 동공을 1/2 이'),
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
