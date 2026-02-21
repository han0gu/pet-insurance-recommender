from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 변경된 보험수익자가 회사에 권리를 대항하기 위해서 계약자는 보험</p><br><table id='9' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>수익자가 변경되었음을</td><td>회사에 "
 '통지하여야 합니다.</td></tr><tr><td colspan="2">부 가 설 명 계약자가 보험수익자가 변경되었음을 회사에 통지하기 '
 '전에 보험금 지급사유가 발생한 경우 회사는 변경 전 보험수익자에게 보험금을 지급할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000194',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
