from langchain_core.documents import Document

chunk = Document(
    page_content=('면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.\n'
 '1. 보험종목 2. 보험기간 3. 보험료 납입주기, 납입방법 및 납입기간 4. 계약자, 피보험자 5. 보험가입금액, 적립보험료 등 기타 '
 '계약의 내용\n'
 '② 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않습 니다. 다만, 변경된 보험수익자가 회사에 권리를 '
 '대항하기 위해서는 계약자가 보험수 익자가 변경되었음을 회사에 통지하여야 합니다.\n'
 '<유의사항>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
