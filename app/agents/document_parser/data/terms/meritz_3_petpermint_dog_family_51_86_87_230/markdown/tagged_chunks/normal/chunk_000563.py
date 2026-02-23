from langchain_core.documents import Document

chunk = Document(
    page_content=('| FGA006 | 상공막염 |  |  |\n'
 '| FGA007 | 녹내장 |  |  |\n'
 '| FGA008 | 고양이 호산구성 각결막염 |  |  |\n'
 '| QBA001 | 눈곱 (원인 불명) |  |  |\n'
 '| QBA002 QBA003 | 결막 충혈 (원인 불명) 눈 가려움증 (원인 불명) |  |  |\n'
 '| 3 | 순환기 질환 | ACA001 | 순환기 계통의 양성 신생물 |\n'
 '| 3 | 순환기 질환 | ACB001 | 순환기 계통의 악성 신생물 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'skin']},
 'indexing': {'chunk_id': 'chunk_000563',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
