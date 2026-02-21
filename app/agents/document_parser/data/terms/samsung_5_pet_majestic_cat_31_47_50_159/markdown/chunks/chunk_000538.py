from langchain_core.documents import Document

chunk = Document(
    page_content=('- 및 반려동물의 연령 증가 등의 사유로 보험요율이 변동될 수 있으며 이 때의 보험료\n'
 '- 는 「보험료 및 해약환급금 산출방법서」 에 따라 산출합니다. 또한, 보험계약의 연장은\n'
 '- 기본계약의 보험기간 내에서만 가능합니다.\n'
 '- ⑥ 제5항에 따라 보험계약이 연장된 경우 계약자는 그 최초연장된 날로부터 90일 이내에\n'
 '- 그 계약을 취소할 수 있으며, 계약자가 연장된 보험계약을 취소하는 경우 회사는 최초\n'
 '- 연장된 날 이후 계약자가 납입한 보험료 전액을 환급합니다.'),
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
