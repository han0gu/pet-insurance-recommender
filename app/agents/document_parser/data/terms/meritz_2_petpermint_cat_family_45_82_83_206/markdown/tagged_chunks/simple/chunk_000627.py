from langchain_core.documents import Document

chunk = Document(
    page_content=('일, 배변, 배뇨는 독립적으로 가능하나 대소변후 뒤처리에 있어 다른 사람의 도움이 필요한 상태(10%) - 빈번하고 불규칙한 배변으로 '
 '인해 2시간 이상 계속되는 업무를 수행하는 것이 어려운 상태 또는 배변, 배뇨는 독립적으로 가능하나 요실 금, 변실금이 있는 때(5%) '
 '|'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000627',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
