from langchain_core.documents import Document

chunk = Document(
    page_content=('거부 등<br>계약자, 피보험자 또는 보험수익자의 책임있는 사유로<br>인하여 보험금 지급사유의 조사 및 확인이 '
 "지연되는<br>경우<br>⑥ 보험금 지급사유에 대해 제3자의 의견에 따르기로 한<br>경우</p><br><h1 id='62' "
 "style='font-size:20px'>【분쟁조정 신청】</h1><br><p id='63' "
 "data-category='paragraph' style='font-size:16px'>분쟁조정 신청은 이 약관의「분쟁의 조정」조항에 "
 '따르<br>며 분쟁조정 신청 대상기관은 금융감독원의'),
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
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
