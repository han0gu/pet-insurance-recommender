from langchain_core.documents import Document

chunk = Document(
    page_content=('기간(이하 「부<br>담보 기간」이라 합니다)은 특정질병의 상태에 따라「1개월<br>부터 5년」또는 「계약의 보험기간」(단, 계약이 갱신 '
 '또는<br>재가입 계약인 경우 최초 계약일로부터 최종 갱신 또는 재<br>가입 계약의 종료일까지의 기간을 말하며, 이하 '
 '「계약의<br>보험기간」이라 합니다)으로 하며, 그 판단기준은 회사에서<br>정한 계약사정기준을 따릅니다'),
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
 'indexing': {'chunk_id': 'chunk_000835',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
