from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 경우 승낙을 서면 등으로 알리거나 보험증권<br>의 뒷면에 기재하여 드립니다.</p><br><p id='14' "
 "data-category='list' style='font-size:20px'>① 보험종목<br>② 보험기간<br>③ 보험료 납입주기, "
 "납입방법 및 납입기간<br>④ 계약자, 피보험자<br>⑤ 보험가입금액, 보험료 등 기타 계약의 내용</p><br><p id='15' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자는 보험수익자를 변경할 수 "
 '있으며 이'),
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
 'indexing': {'chunk_id': 'chunk_000159',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
