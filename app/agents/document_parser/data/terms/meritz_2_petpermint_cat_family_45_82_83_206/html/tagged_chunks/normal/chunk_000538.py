from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 연간 지급하는 총 보험금은<br>보험증권에 기재된 연간 총 보상한도액(1,500만원)을 한도<br>로 '
 "합니다.</p><br><h1 id='64' style='font-size:18px'>【수의사법 제2조(정의)】</h1><br><p "
 "id='65' data-category='paragraph' style='font-size:18px'>이 법에서 사용하는 용어의 뜻은 "
 "다음과 같다.</p><br><p id='66' data-category='list' style='font-size:16px'>1"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000538',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
