from langchain_core.documents import Document

chunk = Document(
    page_content=('주) 1. 만기환급금은 회사가 보험금의 지급시기 도래 7일 이전에 지급할 사유와 금액을 알리지 않은 경우, 지급사유가 발생한 날의 다음 '
 '날부터 청구일까지의 기간은 공시이율을 적용한 이자를 지급합니다. 2. 지급이자의 계산은 연단위 복리로 계산하며, 금리연동형보험은 일자 '
 '계산합니다. 3. 계약자 등의 책임 있는 사유로 보험금 지급이 지연된 때에는 그 해당기간에 대한 이자는 지급 되지 않을 수 '
 '있습니다.다만, 회사는 계약자 등이 분쟁조정을 신청했다는 사유만으로 이자지 급을 거절하지 않습니다. 4'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000859',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
