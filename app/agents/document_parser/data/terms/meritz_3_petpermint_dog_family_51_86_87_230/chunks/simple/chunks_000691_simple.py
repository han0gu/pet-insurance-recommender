from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 지급이자의 계산은 연단위 복리로 계산하며, 일 자 계산합니다. 3. 계약자 등의 책임 있는 사유로 보험금 지급이 지연된 때에는 그 '
 '해당기간에 대한 이자는 지급 되지 않을 수 있습니다. 다만, 회사는 계약자 등이 분쟁조정을 신청했다는 사유만으로 이자지 급을 거절하지 '
 '않습니다. 4. 가산이율 적용시 제8조(보험금의 지급절차) 제2 항 각 호의 어느 하나에 해당되는 사유로 지연 된 경우에는 해당기간에 '
 '대하여 가산이율을 적'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 199},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000691',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
