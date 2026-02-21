from langchain_core.documents import Document

chunk = Document(
    page_content=('지 못한다. 다만, 직접 진료하거나 검안한 수의사가 부득이한 사유로 진단서,\n'
 '검안서 또는 증명서를 발급할 수 없을 때에는 같은 동물병원에 종사하는 다른\n'
 '수의사가 진료부 등에 의하여 발급할 수 있다.\n'
 '② 제1항에 따른 진료 중 폐사(斃死)한 경우에 발급하는 폐사 진단서는 다른 수의사\n'
 '에게서 발급받을 수 있다.\n'
 '③ 수의사는 직접 진료하거나 검안한 동물에 대한 진단서, 검안서, 증명서 또는 처\n'
 '방전의 발급을 요구받았을 때에는 정당한 사유 없이 이를 거부하여서는 아니\n'
 '된다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
