from langchain_core.documents import Document

chunk = Document(
    page_content=('. 지급이자의 계산은 연단위 복리로 계산하며, 일<br>자 계산합니다.<br>3. 계약자 등의 책임 있는 사유로 보험금 '
 '지급이<br>지연된 때에는 그 해당기간에 대한 이자는 지급<br>되지 않을 수 있습니다. 다만, 회사는 계약자<br>등이 분쟁조정을 '
 '신청했다는 사유만으로 이자지<br>급을 거절하지 않습니다.<br>4'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000901',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
