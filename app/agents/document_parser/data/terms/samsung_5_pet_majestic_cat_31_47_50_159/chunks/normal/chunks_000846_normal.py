from langchain_core.documents import Document

chunk = Document(
    page_content=('① 보험계약자(이하 「계약자」 라 합니다)는 제2회 이후의 보험료부터 이 특별약관에 따 라 계약자의 지정계좌를 이용하여 보험료를 '
 '자동납입하거나 급여이체를 통하여 납입 합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 131},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000846',
              'chunk_char_len': 96,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
