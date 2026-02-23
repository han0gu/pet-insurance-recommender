from langchain_core.documents import Document

chunk = Document(
    page_content=('계약의 청약을 철회하는 경우에는 회사는 청약의 철회를 접\n'
 '수한 날부터 3영업일 이내에 해당 신용카드회사로 하여금\n'
 '대금청구를 하지 않도록 해야 하며, 이 경우 회사는 보험료\n'
 '를 반환한 것으로 봅니다.\uf000 청약을 철회할 때에 이미 보험금 지급사유가 발생하였으\n'
 '나 계약자가 그 보험금 지급사유가 발생한 사실을 알지 못\n'
 '한 경우에는 청약철회의 효력은 발생하지 않습니다.\n'
 '\uf000 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
