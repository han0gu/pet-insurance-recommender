from langchain_core.documents import Document

chunk = Document(
    page_content=('7 제1항의 경우 피보험자가 보장개시일(책임개시일) 이후 입원하여 치료를 받던 중 보험\n'
 '기간이 끝났을 때에도 퇴원하기 전까지의 계속중인 입원에 대하여는 제1항에 따라 반# 려견 위탁비용을 계속 보장합니다.⑧ 피보험자가 정당한 '
 '이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사는\n'
 '반려견 위탁비용의 전부 또는 일부를 지급하지 않습니다.# 제2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 '
 '지급사유)의 보험금 지급사유에 대해 합의하지 못'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000679',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
