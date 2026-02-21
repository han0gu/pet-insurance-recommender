from langchain_core.documents import Document

chunk = Document(
    page_content=("id='64' data-category='paragraph' style='font-size:18px'>\uf000 제1항의 통지에 따라 "
 '위험의 증가로 보험료를 더 내야<br>할 경우 회사가 청구한 추가보험료(정산금액을 포함합니다)<br>를 계약자가 납입하지 않았을 때, '
 '회사는 위험이 증가되기<br>전에 적용된 보험요율(이하「변경전 요율」이라 합니다)의<br>위험이 증가된 후에 적용해야 할 '
 '보험요율(이하「변경후 요<br>율」이라 합니다)에 대한 비율에 따라 보험금을 삭감하여<br>지급합니다'),
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
 'indexing': {'chunk_id': 'chunk_000324',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
