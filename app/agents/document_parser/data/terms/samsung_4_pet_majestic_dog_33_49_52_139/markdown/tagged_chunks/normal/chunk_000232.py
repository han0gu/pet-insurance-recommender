from langchain_core.documents import Document

chunk = Document(
    page_content=('는 보험수익자의 권리가 확정됩니다. 그러나 계약자가 사망한 이후 그 승계인이 보험\n'
 '수익자를 변경할 수 있다는 별도의 약정이 있는 경우에는 승계받은 계약자가 보험수\n'
 '익자를 변경할 수 있습니다.\n'
 '⑥ 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을\n'
 '드리고, 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니다.- \n'
 '# 제24조 (보험나이 등)- ① 이 약관에서의 피보험자의 나이는 보험나이를 기준으로 합니다. 다만, 제22조(특별약'),
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
 'indexing': {'chunk_id': 'chunk_000232',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
