from langchain_core.documents import Document

chunk = Document(
    page_content=("계속 납입하던 중 상해사망 사고가 발생한 경우</p><br><p id='42' data-category='list' "
 "style='font-size:16px'>∙ 상해사망 가입금액 : 1억원<br>∙ 상해사망 보험요율 : 1급 0.3, 2급 "
 "0.5<br>⇒ 고객이 수령하는 상해사망 보험금 = 1억원 × (0.3<br>÷ 0.5) = 6천만원</p><br><p id='43' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자 또는 피보험자가 고의 또는 "
 '중대한 과실로 제1항<br>각 호의'),
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
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
