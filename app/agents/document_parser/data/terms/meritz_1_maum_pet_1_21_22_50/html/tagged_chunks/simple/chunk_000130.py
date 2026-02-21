from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험료 납입주기, 납입방법 및 납입기간<br>4. 계약자, 피보험자 중 일부<br>5. 보험가입금액, 보험료, 배상책임의 경우 '
 "보상한도액 등 기타 계약의 내용</p><br><p id='33' data-category='paragraph' "
 "style='font-size:14px'>② 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않습니<br>다"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
