from langchain_core.documents import Document

chunk = Document(
    page_content=('- 자적 의사표시 포함)가 보험계약자 또는 그의\n'
 '- 대리인에게 도달한 날로 봅니다.\n'
 '175【별표2】# 장해분류표# \uf000 총칙# 1. 장해의 정의- 1) “장해”라 함은 상해 또는 질병에 대하여 치유된 후\n'
 '- 신체에 남아있는 영구적인 정신 또는 육체의 훼손상태\n'
 '- 및 기능상실 상태를 말한다. 다만, 질병과 부상의 주\n'
 '- 증상과 합병증상 및 이에 대한 치료를 받는 과정에서\n'
 '- 일시적으로 나타나는 증상은 장해에 포함되지 않는다.\n'
 '- 2) “영구적”이라 함은 원칙적으로 치유하는 때 장래 회'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000510',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
