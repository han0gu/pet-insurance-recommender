from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자, 피보험자 및 보험수익자가 동일한 계약의 경우 2. 계약자, 피보험자가 동일하고 보험수익자가 계약자의 법정상속인인 계약일 '
 '경우\n'
 '⑤ 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 ‘보험개발원이 '
 '공시하는 보험계약대출이율’을 연 단위 복리로 계산한 금액을 더하여 지급합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000084',
              'chunk_char_len': 198,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
