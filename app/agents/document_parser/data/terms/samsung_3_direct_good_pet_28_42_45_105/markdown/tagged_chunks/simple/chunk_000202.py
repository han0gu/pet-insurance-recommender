from langchain_core.documents import Document

chunk = Document(
    page_content=('- 내용으로 변경합니다. 이 경우 승낙을 서면 등으로 알리거나 보험증권의 뒷면에 기재\n'
 '- 하여 드립니다.\n'
 '- ② 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않습\n'
 '- 니다. 다만, 변경된 보험수익자가 회사에 권리를 대항하기 위해서는 계약자가 보험수\n'
 '- 익자가 변경되었음을 회사에 통지하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000202',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
