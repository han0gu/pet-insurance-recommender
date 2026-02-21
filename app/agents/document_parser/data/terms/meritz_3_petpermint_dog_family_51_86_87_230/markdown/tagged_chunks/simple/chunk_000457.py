from langchain_core.documents import Document

chunk = Document(
    page_content=('항의 연간 두번째 이상 지급한도 내에서 보험금을 지급합니\n'
 '다. 단, 동일한 날에 2회 이상의 MRI,CT 및 내시경처치를\n'
 '받은 경우 이를 1회로 보아 제2항의 지급한도 내에서 지급\n'
 '합니다.\n'
 '\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여\n'
 '30일 이내에 발생한 질병은 보상하지 않습니다. 단,「반려\n'
 '동물 비용손해 관련 특별약관 일반조항」제15조(재가입) 제\n'
 '6항에 따라 보험계약이 연장된 경우에는 적용하지 않습니\n'
 '다.168\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000457',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
