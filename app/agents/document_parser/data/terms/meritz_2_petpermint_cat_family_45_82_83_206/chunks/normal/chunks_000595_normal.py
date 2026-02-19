from langchain_core.documents import Document

chunk = Document(
    page_content=('ACB001 | 순환기 계통의 악성 신생물\n'
 'ACC001 | 순환기 계통의 신생물(양성 또는 악성이 불확 실한)\n'
 'HAA001 | 고혈압\n'
 'HAA002 HAA003 | 저혈압 부정맥\n'
 'HAA004 | 판막 질환(의증, 심잡음 포함)\n'
 'HAA005 | 판막 질환 (심부전 증상)\n'
 'HAA006 | 심비대 (원인 불명)\n'
 'HAA007 | 확장성 심근병증\n'
 'HAA008 | 비대성 심근병증\n'
 'HAA009 | 제한성 심근병증\n'
 'HAA010 | 일시적 심근비대증\n'
 'HAA011 | 기타 심근증\n'
 'HAA012 | 대동맥 협착증 · AS'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 170},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000595',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
