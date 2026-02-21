from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 동일한 날에 2회 이상의 MRI,CT 및 내시경처치를<br>받은 경우 이를 1회로 보아 제2항의 지급한도 내에서 '
 '지급<br>합니다.<br>\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여<br>30일 이내에 발생한 질병은 보상하지 '
 '않습니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000674',
              'chunk_char_len': 142,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
