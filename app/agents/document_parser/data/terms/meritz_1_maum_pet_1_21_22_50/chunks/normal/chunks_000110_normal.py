from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 회사는 제1항에 따른 계약자 명의변경 신청 및 계약의 특별부활(효력회복) 청약을 승낙 하며, 계약은 청약한 때부터 '
 '특별부활(효력회복) 됩니다. ③ 회사는 제1항의 통지를 지정된 보험수익자에게 하여야 합니다. 다만, 회사는 법정상속 인이 보험수익자로 '
 '지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다. ④ 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
