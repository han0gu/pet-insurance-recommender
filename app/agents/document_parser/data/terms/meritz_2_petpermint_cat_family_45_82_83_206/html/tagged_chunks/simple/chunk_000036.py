from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>51</footer><p id='48' data-category='paragraph' "
 "style='font-size:20px'>(보험금의 지급사유)의 상해 관련 보험금 지급사유가 발생<br>한 때에는 해당 보험금을 지급하지 "
 "않습니다.</p><br><p id='49' data-category='list' style='font-size:16px'>① "
 '전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙<br>벽을 오르내리거나 특수한 기술, 경험, 사전훈련을 필<br>요로 하는 등반을 '
 '말합니다),'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['urinary']},
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
