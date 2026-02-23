from langchain_core.documents import Document

chunk = Document(
    page_content=('다는 사유만으로 이자지급을 거절하지 않습니다.\n'
 '3. 가산이율 적용시 제9조(보험금의 지급절차) 제2항 각 호의 어느 하나에 해당되는 사\n'
 '유로 지연된 경우에는 해당기간에 대하여 가산이율을 적용하지 않습니다.\n'
 '4. 가산이율 적용시 금융위원회 또는 금융감독원이 정당한 사유로 인정하는 경우에는\n'
 '해당 기간에 대하여 가산이율을 적용하지 않습니다.\n'
 '5. 보험계약대출이율은 보험개발원이 공시하는 보험계약대출이율을 적용합니다.- 48 -<부표2> 보험금을 지급할 때의 적립이율(배상책임 '
 '특별약관 제7조 제2항 관련)| 기 간 | 지 급 이 자 |'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
