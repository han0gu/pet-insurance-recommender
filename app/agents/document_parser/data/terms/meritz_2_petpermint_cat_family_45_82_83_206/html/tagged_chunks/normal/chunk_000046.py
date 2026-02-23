from langchain_core.documents import Document

chunk = Document(
    page_content=('신청은 이 약관의「분쟁의 조정」조항에 따르<br>며 분쟁조정 신청 대상기관은 금융감독원의 금융분쟁조<br>정위원회를 '
 "말합니다.</p><br><p id='64' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제2항에 따라 장해지급률의 판정 및 지급할 보험금의 결<br>정과 관련하여 확정된 "
 '장해지급률에 따른 보험금을 초과한<br>부분에 대한 분쟁으로 보험금 지급이 늦어지는 경우에는 보<br>험수익자의 청구에 따라 이미 확정된 '
 '보험금을 먼저 가지급<br>합니다.<br>\uf000 제2항에 따라'),
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
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
