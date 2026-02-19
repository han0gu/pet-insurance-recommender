from langchain_core.documents import Document

chunk = Document(
    page_content=('주) 능력장애측정기준의 항목 : ㉮ 적절한 음식섭취, ㉯ 대소변관리, 세면, 목욕, 청소 등의 청결 유지, ㉰ 적절한 대화기술 및 '
 '협조적인 대인관계, ㉱ 규칙적인 통원 + 약물 복용, ㉲ 소지품 및 금전관리나 적절한 구매행 위, ㉳ 대중교통이나 일반공공시설의 이용'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 148},
 'term_type': 'special',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'urinary', 'other']},
 'indexing': {'chunk_id': 'chunk_000973',
              'chunk_char_len': 147,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
