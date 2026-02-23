from langchain_core.documents import Document

chunk = Document(
    page_content=('제공하여야 합니다.<br>\uf000 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제<br>공 사실에 관하여 계약자와 회사간에 '
 '다툼이 있는 경우에는<br>회사가 이를 증명하여야 합니다.<br>\uf000 보험설계사 등이 모집과정에서 사용한 회사 제작의 '
 '보험<br>안내자료의 내용이 약관의 내용과 다른 경우에는 계약자에<br>게 유리한 내용으로 계약이 성립된 것으로 '
 "봅니다.</p><br><h1 id='42' style='font-size:20px'>【보험안내자료】</h1><br><p id='43' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000247',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
