from langchain_core.documents import Document

chunk = Document(
    page_content=('QBA002 QBA003 | 결막 충혈 (원인 불명) 눈 가려움증 (원인 불명)\n'
 '3 | 순환기 질환 | ACA001 | 순환기 계통의 양성 신생물\n'
 'ACB001 | 순환기 계통의 악성 신생물\n'
 'ACC001 | 순환기 계통의 신생물(양성 또는 악성이 불확실한)\n'
 'HAA001 | 고혈압\n'
 'HAA002 | 저혈압\n'
 'HAA003 HAA004 | 부정맥\n'
 '판막 질환(의증, 심잡음 포함)\n'
 'HAA005 | 판막 질환 (심부전 증상)\n'
 'HAA006 | 심비대 (원인 불명)\n'
 'HAA007 | 확장성 심근병증\n'
 'HAA008 | 비대성 심근병증'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 196},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['eye', 'other']},
 'indexing': {'chunk_id': 'chunk_000678',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
