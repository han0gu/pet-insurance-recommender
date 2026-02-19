from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[수의사법 제12조(진단서 등)]\n'
 '① 수의사는 자기가 직접 진료하거나 검안하지 아니하고는 진단서, 검안서, 증명서 또는 처방전(「 전자서명법」에 따른 전자서명이 기재된 '
 '전자문서 형태로 작성한 처방전을 포함한다. 이하 같 다 )을 발급하지 못하며, 「약사법」 제85조제6항에 따른 동물용 의약품(이하 '
 '"처방대상 동물용 의약품"이라 한다)을 처방·투약하지 못한다. 다만, 직접 진료하거나 검안한 수의사가 부득이한'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 116},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000714',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
