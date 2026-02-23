from langchain_core.documents import Document

chunk = Document(
    page_content=('다고 인정된 경우로서 수의사의 관리 하에 자기공명영상\n'
 '(MRI)을 사용하는 촬영 의료행위를 말합니다.\uf000 제1항의 전산화단층촬영(CT)이라 함은 제1조(보험금의\n'
 '지급사유)에서 정한 수의사에 의하여 진단 및 치료가 필요\n'
 '하다고 인정된 경우로서 수의사의 관리 하에 전산화단층촬\n'
 '영(CT)을 사용하는 촬영 의료행위를 말합니다.\uf000 제1항의 내시경처치라 함은 제1조(보험금의 지급사유)에\n'
 '서 정한 수의사에 의하여 진단 및 치료가 필요하다고 인정\n'
 '된 경우로서 수의사의 관리 하에 내시경을 이용하여 비침습'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000473',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
