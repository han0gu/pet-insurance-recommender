from langchain_core.documents import Document

chunk = Document(
    page_content=('- 전( 「전자서명법」 에 따른 전자서명이 기재된 전자문서 형태로 작성한 처방전을 포함한다.\n'
 '- 이하 같다) 을 발급하지 못하며, 「약사법」 제85조제6항에 따른 동물용 의약품(이하 "처방\n'
 '- 대상 동물용 의약품"이라 한다)을 처방 · 투약하지 못한다. 다만, 직접 진료하거나 검안한\n'
 '- 수의사가 부득이한 사유로 진단서, 검안서 또는 증명서를 발급할 수 없을 때에는 같은 동물\n'
 '- 병원에 종사하는 다른 수의사가 진료부 등에 의하여 발급할 수 있다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000483',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
