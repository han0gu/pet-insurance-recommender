from langchain_core.documents import Document

chunk = Document(
    page_content=('- 결정한다.\n'
 '- 2) "씹어먹는 기능에 심한 장해를 남긴 때" 라 함은 심한 개구운동 제한이나 저작\n'
 '- 운동 제한으로 물이나 이에 준하는 음료 이외는 섭취하지 못하는 경우를 말한\n'
 '- 다.\n'
 '- 3) "씹어먹는 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상\n'
 '- 138 -# 에 해당되는 때를 말한다.- 가) 뚜렷한 개구운동 제한 또는 뚜렷한 저작운동 제한으로 미음 또는 이에 준하\n'
 '- 는 정도의 음식물(죽 등) 이외는 섭취하지 못하는 경우'),
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
