from langchain_core.documents import Document

chunk = Document(
    page_content=('| 3 | 순환기 질환 | HAA005 | 판막 질환 (심부전 증상) |\n'
 '| 3 | 순환기 질환 | HAA006 | 심비대 (원인 불명) |\n'
 '| 3 | 순환기 질환 | HAA007 | 확장성 심근병증 |\n'
 '| 3 | 순환기 질환 | HAA008 | 비대성 심근병증 |\n'
 '| 3 | 순환기 질환 | HAA009 | 제한성 심근병증 일시적 심근비대증 |\n'
 '| 3 | 순환기 질환 | HAA010 HAA011 | 기타 심근증 |\n'
 '| 3 | 순환기 질환 | HAA012 | 대동맥 협착증 · AS |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000565',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
