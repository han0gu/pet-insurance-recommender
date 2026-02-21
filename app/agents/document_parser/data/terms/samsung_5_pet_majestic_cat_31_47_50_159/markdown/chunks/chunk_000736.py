from langchain_core.documents import Document

chunk = Document(
    page_content=('| 8) 한 눈에 뚜렷한 시야장해를 남긴 때 | 5 |\n'
 '| 9) 한 눈의 눈꺼풀에 뚜렷한 결손을 남긴 때 | 10 |\n'
 '| 10) 한 눈의 눈꺼풀에 뚜렷한 운동장해를 남긴 때 | 5 |\n'
 '- 나. 장해판정기준\n'
 '1) 시력장해의 경우 공인된 시력검사표에 따라 최소 3회 이상 측정한다.- 136 -- \n'
 '- 2) "교정시력" 이라 함은 안경(콘택트렌즈를 포함한 모든 종류의 시력 교정수단)\n'
 '- 으로 교정한 원거리 최대교정시력을 말한다. 다만, 각막이식술을 받은 환자인\n'
 '- 경우 각막이식술 이전의 시력상태를 기준으로 평가한다.'),
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
