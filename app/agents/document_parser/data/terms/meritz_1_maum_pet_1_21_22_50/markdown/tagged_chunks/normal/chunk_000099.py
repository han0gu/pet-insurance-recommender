from langchain_core.documents import Document

chunk = Document(
    page_content=('- 인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다.\n'
 '- ④ 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다. 다만, 회사의\n'
 '- 통지가 7일을 지나서 도달하고 이후 보험수익자가 제1항에 의한 계약자 명의변경 신청\n'
 '- 및 계약의 특별부활(효력회복)을 청약한 경우에는 계약이 해지된 날부터 7일이 되는 날\n'
 '- 에 특별부활(효력회복) 됩니다.\n'
 '- ⑤ 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게 통지된 경우에는 계약자가 통지'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
