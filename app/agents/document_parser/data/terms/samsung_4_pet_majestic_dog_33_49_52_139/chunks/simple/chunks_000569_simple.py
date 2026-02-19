from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항에 따른 진료 중 폐사(斃死)한 경우에 발급하는 폐사 진단서는 다른 수의사에게서 발 급받을 수 있다. ③ 수의사는 직접 '
 '진료하거나 검안한 동물에 대한 진단서, 검안서, 증명서 또는 처방전의 발급 을 요구받았을 때에는 정당한 사유 없이 이를 거부하여서는 '
 '아니된다. ④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서, 증명서 또는 처방전의 서식, 기재사항, 그 밖에 필요한 사항은 '
 '농림축산식품부령으로 정한다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000569',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
