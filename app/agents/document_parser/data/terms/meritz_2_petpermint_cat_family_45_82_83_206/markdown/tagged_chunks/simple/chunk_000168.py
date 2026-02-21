from langchain_core.documents import Document

chunk = Document(
    page_content=('가 청약서에서 질문한 중요한 사항에 대해 사실대로 알\n'
 '려야 하며, 위반하는 경우 계약의 해지 또는 보험금 부\n'
 '지급 등 불이익을 당할 수 있습니다.【상법 제651조(고지의무위반으로 인한 계약해지)】보험계약당시에 보험계약자 또는 피보험자가 고의 '
 '또는\n'
 '중대한 과실로 인하여 중요한 사항을 고지하지 아니하거\n'
 '나 부실의 고지를 한 때에는 보험자는 그 사실을 안 날\n'
 '로부터 1월내에, 계약을 체결한 날로부터 3년내에 한하\n'
 '여 계약을 해지할 수 있다. 그러나 보험자가 계약당시에\n'
 '그 사실을 알았거나 중대한 과실로 인하여 알지 못한 때'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000168',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
