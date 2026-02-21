from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의약품"이라 한다)을 처방·투약하지 못한다. 다만, 직접 진료하거나 검안한 수의사가 부득이한\n'
 '- 사유로 진단서, 검안서 또는 증명서를 발급할 수 없을 때에는 같은 동물병원에 종사하는 다른\n'
 '- 수의사가 진료부 등에 의하여 발급할 수 있다.\n'
 '- ② 제1항에 따른 진료 중 폐사(斃死)한 경우에 발급하는 폐사 진단서는 다른 수의사에게서 발급받\n'
 '- 115 -- 을 수 있다.\n'
 '- ③ 수의사는 직접 진료하거나 검안한 동물에 대한 진단서, 검안서, 증명서 또는 처방전의 발급을'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000615',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
