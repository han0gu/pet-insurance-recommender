from langchain_core.documents import Document

chunk = Document(
    page_content=('| 만기환급금, 및 해약환급금 | 청구일의 다음 날부터 지급일까지 의 기간 | 보험계약대출이율 |\n'
 '- 주) 1. 만기환급금은 회사가 보험금의 지급시기 도래 7일 이전에 지급할 사유와 금액을 알리지 않은\n'
 '- 경우, 지급사유가 발생한 날의 다음 날부터 청구일까지의 기간은 공시이율을 적용한 이자를\n'
 '- 지급합니다.\n'
 '- 2. 지급이자의 계산은 연단위 복리로 계산하며, 금리연동형보험은 일자 계산합니다.\n'
 '- 3. 계약자 등의 책임 있는 사유로 보험금 지급이 지연된 때에는 그 해당기간에 대한 이자는 지급'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000725',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
