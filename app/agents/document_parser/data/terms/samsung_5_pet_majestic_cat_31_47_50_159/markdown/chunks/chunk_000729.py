from langchain_core.documents import Document

chunk = Document(
    page_content=('- 또한 그 증상이 고정된 상태를 말한다.\n'
 '- 라. 다만, 영구히 고정된 증상은 아니지만 치료 종결 후 한시적으로 나타나는 장해에\n'
 '- 대하여는 그 기간이 5년 이상인 경우 해당 장해지급률의 20%를 장해지급률로 한\n'
 '- 다.\n'
 '- 마. 위 라.에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의\n'
 '- 효력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일 또는 질병\n'
 '- 의 진단확정일부터 2년 이내로 하고, 보험기간이 10년 미만인 계약은 상해 발생'),
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
