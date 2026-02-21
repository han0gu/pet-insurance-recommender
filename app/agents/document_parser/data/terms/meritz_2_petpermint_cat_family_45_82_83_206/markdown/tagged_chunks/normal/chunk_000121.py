from langchain_core.documents import Document

chunk = Document(
    page_content=('지할 수 있습니다.\n'
 '\uf000 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후\n'
 '3개월이 지난 때에는 그 효력을 잃습니다.\n'
 '\uf000 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에\n'
 '따라 계약이 효력을 잃는 경우에 회사는 제35조(해약환급\n'
 '금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.# 제35조(해약환급금)\uf000 이 약관에 따른 해약환급금은「보험료 및 '
 '해약환급금 산\n'
 '출방법서」에 따라 계산합니다.\n'
 '\uf000 해약환급금의 지급사유가 발생한 경우 계약자는 회사에\n'
 '해약환급금을 청구하여야 하며, 회사는 청구를 접수한 날부'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
