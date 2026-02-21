from langchain_core.documents import Document

chunk = Document(
    page_content=('내 신생물(양성 또는 악성이 '
 '불확실한)</td></tr><tr><td>JAA001</td><td>치수염</td></tr><tr><td>JAA002</td><td>치아 '
 '골절</td></tr><tr><td>JAA003</td><td>애나멜 '
 '저형성증</td></tr><tr><td>JAA004</td><td>유치 '
 '잔존증</td></tr><tr><td>JAA005</td><td>부정 교합</td></tr><tr><td>JAA006</td><td>기타 '
 '치과 질환</td></tr><tr><td>JBA001</td><td>구내염 /'),
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
 'indexing': {'chunk_id': 'chunk_000892',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
