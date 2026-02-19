from langchain_core.documents import Document

chunk = Document(
    page_content=('JAA006 | 기타 치과 질환\n'
 'JBA001 | 구내염 / 설염\n'
 'JBA002 | 구개열\n'
 'JBA003 | 침샘 질환 (침샘염 / 점액 낭종 / 하마종)\n'
 'JBA004 | 치은염 / 치주염\n'
 'JBA005 JBA006 | 치근농양 / 근첨농양\n'
 'JBA007 | 기타 구강 질환 치아흡수성병변(FORL)\n'
 'JBA008 | 고양이 만성 구내염(FCGs)\n'
 '8 | 전신성 질환\n'
 'PAA018 | 고양이 전염성복막염(FIP)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 173},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'other']},
 'indexing': {'chunk_id': 'chunk_000611',
              'chunk_char_len': 224,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
