from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다.\n'
 '- 2. 만 15세 미만자, 심신상실자 또는 심신박약자를 피보험자로 하여 사망을 보험금 지\n'
 '- 급사유로 한 경우. 다만, 심신박약자가 계약을 체결하거나 소속 단체의 규약에 따\n'
 '- 라 단체보험의 피보험자가 될 때에 의사능력이 있는 경우 계약이 유효합니다.\n'
 '# <용어풀이># [심신상실자(心神喪失者)]의식은 있으나 장애의 정도가 심하여 자신의 행위 결과를 합리적으로 판단할 능력을 갖지 못한 사'),
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
