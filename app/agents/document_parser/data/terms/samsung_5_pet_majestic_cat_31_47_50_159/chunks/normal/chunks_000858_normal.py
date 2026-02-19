from langchain_core.documents import Document

chunk = Document(
    page_content=('지급기일의 91일 이후 기간 | 보험계약대출이율 +가산이율(8.0%)\n'
 '만기환급금, 및 해약환급금 | 지급사유가 발생한 날의 다음날부 터 청구일까지의 기간 | 1년이내 : 공시이율의 50%\n'
 '1년초과 : 공시이율의 40%\n'
 '청구일의 다음 날부터 지급일까지 의 기간 | 보험계약대출이율'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000858',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
