from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 변경된 보험수익자가 회사에 권리를 대항하기 위해서는 계약자가 보험수익자<br>가 변경되었음을 회사에 통지하여야 '
 "합니다.</p><br><h1 id='34' style='font-size:14px'>【설명】</h1><br><p id='35' "
 "data-category='paragraph' style='font-size:14px'>계약자가 보험수익자가 변경되었음을 회사에 통지하기 "
 '전에 보험금 지급사유가 발생<br>한 경우 회사는 변경 전 보험수익자에게 보험금을 지급할 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
