from langchain_core.documents import Document

chunk = Document(
    page_content=('나) 머리뼈의 손바닥 크기 이상의 손상 및 결손\n'
 '3) 목\n'
 '손바닥 크기 이상의 추상(추한 모습)\n'
 '라. 약간의 추상(추한 모습)\n'
 '1) 얼굴\n'
 '가) 손바닥 크기 1/4 이상의 추상(추한 모습) 나) 길이 5cm 이상의 추상반흔(추한 모습의 흉터) 다) 지름 2cm 이상의 조직함몰 '
 '라) 코의 1/4이상 결손\n'
 '2) 머리\n'
 '가) 손바닥 크기 1/2 이상의 반흔(흉터) 및 모발결손 나) 머리뼈의 손바닥 크기 1/2 이상의 손상 및 결손\n'
 '3) 목\n'
 '손바닥 크기 1/2 이상의 추상(추한 모습)\n'
 '마. 손바닥 크기'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 210},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000738',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
