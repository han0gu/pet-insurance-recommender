from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7 | 치아 및 구강 질환 | JBA002 | 구개열 |\n'
 '| 7 | 치아 및 구강 질환 | JBA003 | 침샘 질환 (침샘염 / 점액 낭종 / 하마종) |\n'
 '| 7 | 치아 및 구강 질환 | JBA004 | 치은염 / 치주염 |\n'
 '| 7 | 치아 및 구강 질환 | JBA005 | 치근농양 / 근첨농양 |\n'
 '| 7 | 치아 및 구강 질환 | JBA006 | 기타 구강 질환 |\n'
 '| 7 | 치아 및 구강 질환 | JBA007 JBA008 | 치아흡수성병변(FORL) 고양이 만성 구내염(FCGs) |\n'
 '| 8 | 전신성 질환 |  |  |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000503',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
