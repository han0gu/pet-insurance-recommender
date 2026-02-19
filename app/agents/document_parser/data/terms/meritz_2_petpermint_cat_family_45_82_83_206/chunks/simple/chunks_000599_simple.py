from langchain_core.documents import Document

chunk = Document(
    page_content=('OAA008 | 요도 폐색\n'
 'OAA009 | 요로 결석증\n'
 'OAA010 | 신경성 배뇨 이상\n'
 'OAA011 | 고양이 특발성 방광염(FIC)\n'
 'OAA012 | 고양이 하부 비뇨기계 질환(FLUTD)\n'
 'OAA013 | 고양이 하부 요로계 증후군(FUS)\n'
 'OAA014 | 기타 비뇨기계 질환\n'
 'OAA015 | 다낭성 신장 질환\n'
 'OAA016 | 단백 소실성 신증(PLN)\n'
 'QGA001 | 혈뇨 (원인 불명)\n'
 'QGA002 | 요실금 (원인 불명)\n'
 'QGA003 QGA004 | 비정상 성분의 소변 (원인 불명) 핍뇨 (원인 불명)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 171},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['urinary']},
 'indexing': {'chunk_id': 'chunk_000599',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
