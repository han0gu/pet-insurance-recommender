from langchain_core.documents import Document

chunk = Document(
    page_content=('① 수의사는 자기가 직접 진료하거나 검안하지 아니하고는 진단서, 검안서, 증명서 또는 처방 전(「전자서명법」에 따른 전자서명이 기재된 '
 '전자문서 형태로 작성한 처방전을 포함한다. 이하 같다) 을 발급하지 못하며, 「약사법」 제85조제6항에 따른 동물용 의약품(이하 "처방 '
 '대상 동물용 의약품"이라 한다)을 처방·투약하지 못한다. 다만, 직접 진료하거나 검안한 수의사가 부득이한 사유로 진단서, 검안서 또는 '
 '증명서를 발급할 수 없을 때에는 같은 동물 병원에 종사하는 다른 수의사가 진료부 등에 의하여 발급할 수 있다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 70},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000373',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
