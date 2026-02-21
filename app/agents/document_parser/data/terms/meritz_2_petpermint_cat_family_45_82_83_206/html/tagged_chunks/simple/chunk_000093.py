from langchain_core.documents import Document

chunk = Document(
    page_content=('. 한편 위<br>험이 증가된 경우에는 보험료의 증액 및 정산금액의 추가납<br>입을 요구할 수 있으며, 계약자는 일시납 또는 잔여 '
 '보험료<br>납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수<br>없음) 동안의 분납 중 선택하여 정산금액을 납입하여야 '
 '합<br>니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
