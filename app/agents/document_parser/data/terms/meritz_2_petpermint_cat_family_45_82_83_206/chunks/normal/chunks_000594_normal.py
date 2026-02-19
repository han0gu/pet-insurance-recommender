from langchain_core.documents import Document

chunk = Document(
    page_content=('FFA002 | 망막 박리 (유리체 변성 포함)\n'
 'FGA001 | 녹내장 (좌안)\n'
 'FGA002 | 녹내장 (우안)\n'
 'FGA003 | 동양안충증\n'
 'FGA004 | 기타 안과 질환\n'
 'FGA005 | 초자체변성\n'
 'FGA006 | 상공막염\n'
 'FGA007 | 녹내장\n'
 'FGA008 | 고양이 호산구성 각결막염\n'
 'QBA001 | 눈곱 (원인 불명)\n'
 'QBA002 | 결막 충혈 (원인 불명)\n'
 'QBA003 | 눈 가려움증 (원인 불명)\n'
 '3 | 순환기 질환 | ACA001 | 순환기 계통의 양성 신생물\n'
 'ACB001 | 순환기 계통의 악성 신생물'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 170},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000594',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
