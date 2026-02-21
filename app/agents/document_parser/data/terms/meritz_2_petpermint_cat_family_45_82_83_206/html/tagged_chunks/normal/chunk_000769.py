from langchain_core.documents import Document

chunk = Document(
    page_content=("보험증권에 기재된 보상비율(70%)을 곱한 금액을 아래<br>에서 정한 금액을 한도로 보상합니다.</p><br><table id='1' "
 'style=\'font-size:16px\'><thead><tr><td colspan="4">항목</td><td>자기 '
 '부담금</td><td>지급 한도</td></tr></thead><tbody><tr><td rowspan="4">입 원 의 료 비 '
 'Ⅲ</td><td rowspan="3">입원 중 수술을 하지 않은 날의 경우</td><td rowspan="2">MRI,CT 및 '
 '내시경처치 를 받은 날의'),
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
 'indexing': {'chunk_id': 'chunk_000769',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
