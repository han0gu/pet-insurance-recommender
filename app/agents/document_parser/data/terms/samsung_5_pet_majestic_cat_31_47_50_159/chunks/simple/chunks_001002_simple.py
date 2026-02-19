from langchain_core.documents import Document

chunk = Document(
    page_content=('4 . 요추 및 골반의 골절 | S32\n'
 '5 . 대퇴골의 골절 | S72\n'
 '6 . 출산손상으로 인한 두개골골절 | P13.0\n'
 '7 . 두개골의 기타 출산손상 | P13.1\n'
 '8 . 척추 및 척수의 출산손상 | P11.5\n'
 '9 . 대퇴골의 출산손상 | P13.2'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 152},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['joint', 'head', 'other', 'other', 'other']},
 'indexing': {'chunk_id': 'chunk_001002',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
