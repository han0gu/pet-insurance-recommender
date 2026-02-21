from langchain_core.documents import Document

chunk = Document(
    page_content=('료행위가 연간 첫 번째로 발생한 때에는 제2항의 연간 첫\n'
 '번째 지급한도 내에서 보험금을 지급하며 연간 첫 번째 의\n'
 '료행위 이후에 발생한 MRI,CT 및 내시경처치에 대하여 제2\n'
 '항의 연간 두번째 이상 지급한도 내에서 보험금을 지급합니\n'
 '다. 단, 동일한 날에 2회 이상의 MRI,CT 및 내시경처치를\n'
 '받은 경우 이를 1회로 보아 제2항의 지급한도 내에서 지급\n'
 '합니다.\n'
 '\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여\n'
 '30일 이내에 발생한 질병은 보상하지 않습니다. 단,「반려'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000380',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
