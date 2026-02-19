from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약관계 관련 용어\n'
 '1. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다. 2. 보험수익자: 보험금 지급사유가 발생하는 때에 '
 '회사에 보험금을 청구하여 받을 수 있는 사람을 말합니다. 3. 보험증권: 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 '
 '드리는 증 서를 말합니다. 4. 진단계약: 계약을 체결하기 위하여 피보험자가 건강진단을 받아야 하는 계약을 말 합니다. 5. 피보험자: '
 '보험사고의 대상이 되는 사람을 말합니다.\n'
 '② 지급사유 관련 용어'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000178',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
