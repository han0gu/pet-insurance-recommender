from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 지<br>급예정일은 다음 각 호의 어느 하나에 해당하는 경우를 제<br>외하고는 제7조(보험금의 청구)에서 정한 서류를 접수한 '
 "날<br>부터 30영업일 이내에서 정합니다.</p><br><p id='61' data-category='list' "
 "style='font-size:20px'>① 소송제기<br>② 분쟁조정 신청<br>③ 수사기관의 조사<br>④ 해외에서 발생한 보험사고에 "
 '대한 조사<br>⑤ 제6항에 따른 회사의 조사요청에 대한 동의 거부 등<br>계약자, 피보험자 또는 보험수익자의 책임있는 '
 '사유로<br>인하여 보험금'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
