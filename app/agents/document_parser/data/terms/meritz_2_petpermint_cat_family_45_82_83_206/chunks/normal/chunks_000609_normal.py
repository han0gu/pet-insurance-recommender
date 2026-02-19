from langchain_core.documents import Document

chunk = Document(
    page_content=('KGA001 | 트리코모나스증\n'
 'KGA002 | 지아르디아 증\n'
 'KGA003 | 콕시듐증\n'
 'KGA004 | 회충증\n'
 'KGA005 | 촌충증\n'
 'KGA006 | 간충증\n'
 'KGA007 | 기타 소화기계 기생충증\n'
 'KGA008 | 기타 소화기계 감염증\n'
 'KGA009 | 소화계통의 기타 질환\n'
 'PAA014 PAA015 | 고양이 파보 바이러스(FPV) 고양이 코로나 바이러스 감염\n'
 'QEA001 | 구토 (원인 불명)\n'
 'QEA002 | 설사 / 혈변 (원인 불명)\n'
 'QEA003 | 복통 (원인 불명)\n'
 'QEA004 | 복수 (원인 불명)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 173},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000609',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
