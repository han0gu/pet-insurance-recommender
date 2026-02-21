from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관\n'
 '- 험수익자로 변경하여 특별약관의 특별부활(효력회복)을 청약할 수 있음을 보험수\n'
 '- 익자에게 통지하여야 합니다.\n'
 '- \uf000 회사는 제1항에 따른 계약자 명의변경 신청 및 계약의 특별부활(효력회복) 청약\n'
 '- 을 승낙합니다.\n'
 '- \uf000 회사는 제1항의 통지를 지정된 보험수익자에게 하여야 합니다. 다만, 회사는 법정상 상\n'
 '- 속인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다. 해\n'
 '- \uf000 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000524',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
