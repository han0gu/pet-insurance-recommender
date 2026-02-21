from langchain_core.documents import Document

chunk = Document(
    page_content=('- Clinical Dementia Rating) 검사결과에 따른다.\n'
 '# 4) 뇌전증- 가) “뇌전증”이라 함은 돌발적 뇌파이상을 나타내는\n'
 '- 뇌질환으로 발작(경련, 의식장해 등)을 반복하는\n'
 '- 것을 말한다.\n'
 '- 나) 뇌전증 발작의 빈도 및 양상은 지속적인 항뇌전\n'
 '- 증제(항경련제) 약물로도 조절되지 않는 뇌전증\n'
 '- 을 말하며, 진료기록에 기재되어 객관적으로 확\n'
 '- 인되는 뇌전증 발작의 빈도 및 양상을 기준으로\n'
 '228# 한다.- 다) “심한 뇌전증 발작”이라 함은 월 8회 이상의 중'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000693',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
