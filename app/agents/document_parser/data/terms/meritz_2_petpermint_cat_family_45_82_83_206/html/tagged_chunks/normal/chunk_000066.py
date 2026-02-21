from langchain_core.documents import Document

chunk = Document(
    page_content=('. 일시에 지급할 금액을 나누어 지급하는 경우<br>보험금 : 6천만원<br>보험금 지급일자 : 2025년 4월 1일<br>보험금을 '
 "일시에 지급받지 않고, 3년간 매년 동일한 금액<br>으로 나누어 지급받는 경우</p><br><table id='91' "
 "style='font-size:20px'><thead></thead><tbody><tr><td>지급일</td><td>보험금 받는 방법 "
 '변경 후 지급액</td></tr><tr><td>2025년 4월 1일</td><td>2천만원</td></tr><tr><td>2026년 4월'),
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
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
