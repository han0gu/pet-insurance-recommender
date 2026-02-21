from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 보험료 납입주기, 납입방법 및 납입기간\n'
 '4. 계약자, 피보험자 중 일부\n'
 '5. 보험가입금액, 보험료, 배상책임의 경우 보상한도액 등 기타 계약의 내용② 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 '
 '승낙이 필요하지 않습니\n'
 '다. 다만, 변경된 보험수익자가 회사에 권리를 대항하기 위해서는 계약자가 보험수익자\n'
 '가 변경되었음을 회사에 통지하여야 합니다.【설명】계약자가 보험수익자가 변경되었음을 회사에 통지하기 전에 보험금 지급사유가 발생\n'
 '한 경우 회사는 변경 전 보험수익자에게 보험금을 지급할 수 있습니다. 회사가 변경'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000076',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
