from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약관계 관련 용어</h1><br><p id='7' data-category='list' "
 "style='font-size:14px'>가. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다.<br>나. "
 '보험수익자: 보험금 지급사유가 발생하는 때에 회사에 보험금을 청구하여 받을 수<br>있는 사람을 말합니다.<br>다. 보험증권: 계약의 '
 '성립과 그 내용을 증명하기 위하여 회사가 계약자에게 드리는 증<br>서를 말합니다.<br>라'),
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
 'indexing': {'chunk_id': 'chunk_000003',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
