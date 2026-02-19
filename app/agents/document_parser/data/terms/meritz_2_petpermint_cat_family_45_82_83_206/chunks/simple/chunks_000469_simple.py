from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 동일한 날에 2회 이상의 MRI,CT 및 내시경처치를 받은 경우 이를 1회로 보아 제2항의 지급한도 내에서 지급 합니다. '
 '\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여 30일 이내에 발생한 질병은 보상하지 않습니다. 단,「반려 동물 '
 '비용손해 관련 특별약관 일반조항」제15조(재가입) 제 6항에 따라 보험계약이 연장된 경우에는 적용하지 않습니 다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000469',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
