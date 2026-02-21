from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 진단서, 검안서, 증명서 또는 처방전을 발급하지\n'
 '- 못하며, 「약사법」 제85조제6항에 따른 동물용 의약\n'
 '- 품(이하 "동물용 의약품"이라 한다)을 처방·투약하\n'
 '- 지 못한다. 다만, 직접 진료하거나 검안한 수의사가\n'
 '- 부득이한 사유로 진단서, 검안서 또는 증명서를 발급\n'
 '- 할 수 없을 때에는 같은 동물병원에 종사하는 다른\n'
 '- 수의사가 진료부 등에 의하여 발급할 수 있다.\n'
 '- ② 제1항에 따른 진료 중 폐사(斃死)한 경우에 발급하는\n'
 '- 폐사 진단서는 다른 수의사에게서 발급받을 수 있다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
