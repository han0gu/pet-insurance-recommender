from langchain_core.documents import Document

chunk = Document(
    page_content=('검사 소견 | 양측 전정기능 소실 | 14\n'
 '양측 전정기능 감소 | 10\n'
 '일측 전정기능 소실 | 4\n'
 '치료 병력 | 장기 통원치료(1년간 12회이상) | 6\n'
 '장기 통원치료(1년간 6회이상) | 4\n'
 '단기 통원치료(6개월간 6회이상) | 2\n'
 '단기 통원치료(6개월간 6회미만) | 0\n'
 '기능 장해 소견 | 두 눈을 감고 일어서기 곤란하거나 두 눈 을 뜨고 10m 거리를 직선으로 걷다가 쓰 | 20\n'
 '경우 | 12'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 205},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000717',
              'chunk_char_len': 224,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
