from langchain_core.documents import Document

chunk = Document(
    page_content=('| 보장관련 보험금 | 지급기일의 61일 이후부터 90일 이내 기간 | 보험계약대출이율 +가산이율(6.0%) |\n'
 '| 보장관련 보험금 | 지급기일의 91일 이후 기간 | 보험계약대출이율 +가산이율(8.0%) |\n'
 '| 만기환급금, 및 해약환급금 | 지급사유가 발생한 날의 다음날부 터 청구일까지의 기간 | 1년이내 : 공시이율의 50% |\n'
 '| 만기환급금, 및 해약환급금 | 지급사유가 발생한 날의 다음날부 터 청구일까지의 기간 | 1년초과 : 공시이율의 40% |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000724',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
