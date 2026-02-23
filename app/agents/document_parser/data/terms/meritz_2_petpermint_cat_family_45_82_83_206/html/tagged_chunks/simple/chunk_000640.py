from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>\uf000 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생<br>한 때에는 보험금을 "
 "지급하지 않습니다.</p><br><p id='22' data-category='list' style='font-size:16px'>① "
 '계약자, 피보험자, 이들의 가족 또는 사용인의 고의<br>또는 중대한 과실<br>② 지진, 분화, 해일, 홍수 또는 이와 유사한 '
 '자연재해로<br>생긴 손해<br>③ 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소<br>요, 기타 이들과 유사한 사태<br>④ '
 '핵연료물질'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000640',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
