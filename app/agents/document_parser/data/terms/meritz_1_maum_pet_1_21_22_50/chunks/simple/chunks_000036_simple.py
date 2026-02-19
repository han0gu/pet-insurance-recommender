from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항에 따른 진료 중 폐사(斃死)한 경우에 발급하는 폐사 진단서는 다른 수의사 에게서 발급받을 수 있다. ③ 수의사는 직접 '
 '진료하거나 검안한 동물에 대한 진단서, 검안서, 증명서 또는 처 방전의 발급을 요구받았을 때에는 정당한 사유 없이 이를 거부하여서는 아니 '
 '된다. ④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서, 증명서 또는 처방전의 서 식, 기재사항, 그 밖에 필요한 사항은 '
 '농림축산식품부령으로 정한다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
