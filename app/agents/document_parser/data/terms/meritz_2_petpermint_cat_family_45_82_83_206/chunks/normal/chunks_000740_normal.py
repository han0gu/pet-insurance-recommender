from langchain_core.documents import Document

chunk = Document(
    page_content=('※ 주) 능력장애측정기준의 항목 : ㉮ 적절한 음식 섭취, ㉯ 대소변관리, 세면, 목욕, 청소 등의 청 결 유지, ㉰ 적절한 대화기술 및 '
 '협조적인 대인 관계, ㉱ 규칙적인 통원․약물 복용, ㉲ 소지품 및 금전관리나 적절한 구매행위, ㉳ 대중교통이나 일반공공시설의 이용'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 202},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000740',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
