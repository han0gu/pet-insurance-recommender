from langchain_core.documents import Document

chunk = Document(
    page_content=('중 수술을 한 날의 경우</td><td>수술당일에 한하여 1일당 200만원</td></tr></tbody></table><footer '
 "id='7' style='font-size:14px'>147</footer><h1 id='8' "
 "style='font-size:20px'>【보험금 지급금액 산출방식】</h1><br><p id='9' "
 "data-category='paragraph' style='font-size:20px'>보험금 지급금액 = [(피보험자가 부담한 "
 '치료비－자기부담금)<br>× 보상비율, 지급 한도액] 중 적은'),
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
 'indexing': {'chunk_id': 'chunk_000706',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
