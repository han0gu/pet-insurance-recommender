from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험종목 2. 보험기간 3. 보험료 납입주기, 납입방법 및 납입기간 4. 계약자, 피보험자 중 일부 5. 보험가입금액, 보험료, '
 '배상책임의 경우 보상한도액 등 기타 계약의 내용\n'
 '② 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않습니 다. 다만, 변경된 보험수익자가 회사에 권리를 '
 '대항하기 위해서는 계약자가 보험수익자 가 변경되었음을 회사에 통지하여야 합니다.\n'
 '【설명】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
