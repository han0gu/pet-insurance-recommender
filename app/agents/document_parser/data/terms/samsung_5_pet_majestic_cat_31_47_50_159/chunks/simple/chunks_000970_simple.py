from langchain_core.documents import Document

chunk = Document(
    page_content=('5) 정신행동에 약간의 장해를 남긴 때 | 25\n'
 '6) 정신행동에 경미한 장해를 남긴 때 | 10\n'
 '7) 극심한 치매 : CDR 척도 5점 | 100\n'
 '8) 심한 치매 : CDR 척도 4점 | 80\n'
 '9) 뚜렷한 치매 : CDR 척도 3점 | 60\n'
 '10) 약간의 치매 : CDR 척도 2점 | 40\n'
 '11) 심한 뇌전증 발작이 남았을 때 | 70\n'
 '12) 뚜렷한 뇌전증 발작이 남았을 때 | 40\n'
 '13) 약간의 뇌전증 발작이 남았을 때 | 10\n'
 '나. 장해판정기준\n'
 '1) 신경계'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000970',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
