from langchain_core.documents import Document

chunk = Document(
    page_content=('| 3 | 순환기 질환 | ACB001 | 순환기 계통의 악성 신생물 |\n'
 '| 3 | 순환기 질환 | ACC001 | 순환기 계통의 신생물(양성 또는 악성이 불확실한) |\n'
 '| 3 | 순환기 질환 | HAA001 | 고혈압 |\n'
 '| 3 | 순환기 질환 | HAA002 | 저혈압 |\n'
 '| 3 | 순환기 질환 | HAA003 HAA004 | 부정맥 |\n'
 '| 3 | 순환기 질환 |  | 판막 질환(의증, 심잡음 포함) |\n'
 '| 3 | 순환기 질환 | HAA005 | 판막 질환 (심부전 증상) |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000564',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
