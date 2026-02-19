from langchain_core.documents import Document

chunk = Document(
    page_content=('LAA009 | 지루성 피부염\n'
 'LAA010 | 피하 농양\n'
 'LAA011 | 지방층염\n'
 'LAA012 | 호산구성 육아종\n'
 'LAA013 | 홍반루프스\n'
 'LAA014 | 천포창\n'
 'LAA015 | 지간 피부염\n'
 'LAA016 | 족피부염\n'
 'LAA017 | 꼬리샘 과증식\n'
 'LAA018 | 발톱 주위염\n'
 'LAA019 | 옴진드기 · 개선충\n'
 'LAA020 LAA022 | 벼룩 / 진드기 등 외부 기생충 질환 기타 피부 질환\n'
 'QCA001 | 귀 가려움증 (원인 불명)\n'
 'QFA001 | 발진 (원인 불명)\n'
 'QFA002 | 피부염 (원인 불명)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 172},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000604',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
