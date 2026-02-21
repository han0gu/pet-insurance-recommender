from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항에서 정한 대로 계약자 또는 보험수익자가 변경내용을 알리지 않은 경우에는 계\n'
 '약자 또는 보험수익자가 회사에 알린 최종의 주소 또는 연락처로 등기우편 등 우편물\n'
 '에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이\n'
 '지난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.- \n'
 '# 제14조 (보험수익자의 지정)① 보험수익자를 지정하지 않은 때에는 보험수익자를 제11조(만기환급금의 지급) 제1항- 37 -의 경우는 '
 '계약자로 하고, 사망보험금의 경우는 피보험자의 법정상속인, 기타 보험금'),
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
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
