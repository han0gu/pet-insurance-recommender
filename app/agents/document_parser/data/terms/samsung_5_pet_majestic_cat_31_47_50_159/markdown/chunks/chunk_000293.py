from langchain_core.documents import Document

chunk = Document(
    page_content=('않는 경우에는 상해 발생일부터 180일이 되는 날의 의사진단에 기초하여 고정될 것으\n'
 '로 인정되는 상태를 장해지급률로 결정합니다. 다만, 장해분류표에 장해판정시기를 별\n'
 '도로 정한 경우에는 그에 따릅니다.\n'
 '② 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의 효\n'
 '력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일부터 2년 이내로\n'
 '하고, 보험기간이 10년 미만인 계약은 상해 발생일부터 1년 이내)에 장해상태가 더 악\n'
 '화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 결정합니다.'),
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
