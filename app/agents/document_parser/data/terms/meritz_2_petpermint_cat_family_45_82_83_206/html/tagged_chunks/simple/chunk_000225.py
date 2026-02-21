from langchain_core.documents import Document

chunk = Document(
    page_content=('. 해약환급금<br>지급일까지의 기간에 대한 이자의 계산은【별표1(보험금을<br>지급할 때의 적립이율 계산)】에 '
 '따릅니다.<br>\uf000 회사는 경과기간별 해약환급금에 관한 표를 계약자에게<br>제공하여 드립니다.<br>\uf000 '
 '제32조의1(위법계약의 해지)에 따라 위법계약이 해지되<br>는 경우 회사가 적립한 해지 당시의 계약자적립액 및 미경<br>과보험료를 '
 "반환하여 드립니다.</p><h1 id='7' style='font-size:20px'>제36조(보험계약대출)</h1><br><p "
 "id='8'"),
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
 'indexing': {'chunk_id': 'chunk_000225',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
