from langchain_core.documents import Document

chunk = Document(
    page_content=('. 나누어 지급할 금액을 일시에 지급하는 경우<br>보험금 : 매년 1천만원<br>보험금 지급기간 : 3년<br>보험금 지급 시작일자 : '
 '2025년 4월 1일<br>보험금을 3년간 나누어 지급받지 않고, 2025년 4월 1일<br>보험금을 일시에 지급받는 '
 "경우</p><br><table id='93' "
 "style='font-size:20px'><thead></thead><tbody><tr><td>지급일</td><td>보험금 받는 방법 "
 '변경 후 지급액</td></tr><tr><td>2025년 4월 1일</td><td>1천만원 +1천만원'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
