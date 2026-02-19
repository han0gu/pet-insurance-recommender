from langchain_core.documents import Document

chunk = Document(
    page_content=('QEA003 | 복통 (원인 불명)\n'
 'QEA004 | 복수 (원인 불명)\n'
 'QEA005 | 위장관 출혈(혈토, 혈변)\n'
 '7 | 치아 및 구강 질환 | AAA001 | 구강 내 양성 신생물\n'
 'AAB001 | 구강 내 악성 신생물\n'
 'AAC001 | 구강 내 신생물(양성 또는 악성이 불확실한)\n'
 'JAA001 | 치수염\n'
 'JAA002 | 치아 골절\n'
 'JAA003 | 애나멜 저형성증\n'
 'JAA004 | 유치 잔존증\n'
 'JAA005 | 부정 교합\n'
 'JAA006 | 기타 치과 질환\n'
 'JBA001 | 구내염 / 설염\n'
 'JBA002 | 구개열'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 173},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'dental']},
 'indexing': {'chunk_id': 'chunk_000610',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
