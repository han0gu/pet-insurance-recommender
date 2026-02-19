from langchain_core.documents import Document

chunk = Document(
    page_content=('① 수의사는 자기가 직접 진료하거나 검안하지 아니하고 는 진단서, 검안서, 증명서 또는 처방전을 발급하지 못하며, 「약사법」 '
 '제85조제6항에 따른 동물용 의약 품(이하 "동물용 의약품"이라 한다)을 처방·투약하 지 못한다. 다만, 직접 진료하거나 검안한 수의사가 '
 '부득이한 사유로 진단서, 검안서 또는 증명서를 발급 할 수 없을 때에는 같은 동물병원에 종사하는 다른 수의사가 진료부 등에 의하여 발급할 '
 '수 있다. ② 제1항에 따른 진료 중 폐사(斃死)한 경우에 발급하는 폐사 진단서는 다른 수의사에게서 발급받을 수 있다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000197',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
